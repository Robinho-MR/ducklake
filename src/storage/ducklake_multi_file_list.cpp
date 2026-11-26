#include "common/ducklake_util.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_multi_file_list.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_inlined_data_reader.hpp"
#include "storage/ducklake_metadata_manager.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "storage/ducklake_table_entry.hpp"

namespace duckdb {

DuckLakeFileListData::DuckLakeFileListData(DuckLakeFunctionInfo &read_info,
                                           vector<DuckLakeDataFile> transaction_local_files_p,
                                           shared_ptr<DuckLakeInlinedData> transaction_local_data_p,
                                           unique_ptr<FilterPushdownInfo> filter_info_p)
    : read_info(read_info), read_file_list(false), transaction_local_files(std::move(transaction_local_files_p)),
      transaction_local_data(std::move(transaction_local_data_p)), filter_info(std::move(filter_info_p)) {
}

DuckLakeFileListData::DuckLakeFileListData(DuckLakeFunctionInfo &read_info, vector<DuckLakeFileListEntry> files_to_scan)
    : read_info(read_info), files(std::move(files_to_scan)), read_file_list(true) {
}

static DuckLakeFileListEntry InlinedDataEntry(const DuckLakeInlinedTableInfo &inlined_table) {
	DuckLakeFileListEntry file_entry;
	file_entry.file.path = inlined_table.table_name.GetIdentifierName();
	file_entry.row_id_start = 0;
	file_entry.data_type = DuckLakeDataType::INLINED_DATA;
	return file_entry;
}

DuckLakeFileListData::DuckLakeFileListData(DuckLakeFunctionInfo &read_info,
                                           const DuckLakeInlinedTableInfo &inlined_table)
    : read_info(read_info), read_file_list(true) {
	files.push_back(InlinedDataEntry(inlined_table));
	inlined_data_tables.push_back(inlined_table);
}

optional_ptr<const DuckLakeFieldId> DuckLakeFileListData::ResolveFilterField(const Expression &subject,
                                                                             column_t column_id) const {
	if (IsVirtualColumn(column_id)) {
		return nullptr;
	}
	vector<StructExtractPathEntry> path;
	auto &root = PeelStructExtractPath(subject, path);
	if (!ExpressionFilter::IsSimpleFilterColumnRef(root)) {
		return nullptr;
	}
	optional_ptr<const DuckLakeFieldId> field_id = read_info.table.GetFieldId(PhysicalIndex(column_id));
	for (auto &entry : path) {
		if (entry.child_name.empty()) {
			// LCOV_EXCL_START
			return nullptr;
			// LCOV_EXCL_STOP
		}
		field_id = field_id->GetChildByName(entry.child_name);
		if (!field_id) {
			// LCOV_EXCL_START
			return nullptr;
			// LCOV_EXCL_STOP
		}
	}
	return field_id;
}

unique_ptr<DuckLakeFilterNode> DuckLakeFileListData::GetColumnFilterNode(column_t column_id, const Expression &expr,
                                                                         const LogicalType &column_type) const {
	auto subject = DuckLakeUtil::GetFilterSubject(expr);
	if (subject) {
		if (!DuckLakeUtil::IsStructExtract(*subject)) {
			return make_uniq<DuckLakeFilterNode>(
			    ColumnFilterInfo(read_info.table.GetFieldId(PhysicalIndex(column_id)).GetFieldIndex().index,
			                     column_type, make_uniq<ExpressionFilter>(expr.Copy())));
		}
		// stats for a nested field are stored against that field, not against the column that contains it
		auto field_id = ResolveFilterField(*subject, column_id);
		if (!field_id) {
			return nullptr;
		}
		auto rewritten =
		    ExpressionIterator::ReplaceExpression(expr, *subject, BoundReferenceExpression(field_id->Type(), 0U));
		return make_uniq<DuckLakeFilterNode>(ColumnFilterInfo(field_id->GetFieldIndex().index, field_id->Type(),
		                                                      make_uniq<ExpressionFilter>(std::move(rewritten))));
	}
	// a conjunction may constrain several nested fields of the same column, each with their own stats
	if (expr.GetExpressionType() != ExpressionType::CONJUNCTION_AND) {
		return nullptr;
	}
	auto result = make_uniq<DuckLakeFilterNode>(DuckLakeFilterNodeType::CONJUNCTION_AND);
	for (auto &child : expr.Cast<BoundConjunctionExpression>().GetChildren()) {
		auto node = GetColumnFilterNode(column_id, *child, column_type);
		if (node) {
			result->children.push_back(std::move(node));
		}
	}
	if (result->children.empty()) {
		return nullptr;
	}
	return std::move(result);
}

unique_ptr<DuckLakeFilterNode> DuckLakeFileListData::GetFilterNode(column_t column_id,
                                                                   unique_ptr<TableFilter> filter) const {
	if (IsVirtualColumn(column_id)) {
		return nullptr;
	}
	auto column_index = PhysicalIndex(column_id);
	// Get the column type from the table schema, not from the scan types array
	const auto &column_type = read_info.column_types[column_index.index];
	auto expr_filter = ExpressionFilter::FromTableFilter(*filter, column_type);
	if (!expr_filter->expr) {
		// LCOV_EXCL_START
		return nullptr;
		// LCOV_EXCL_STOP
	}
	return GetColumnFilterNode(column_id, *expr_filter->expr, column_type);
}

unique_ptr<DuckLakeFilterNode> DuckLakeFileListData::GetExpressionFilterNode(MultiFilePushdownInfo &info,
                                                                             const Expression &expr) const {
	auto subject = DuckLakeUtil::GetFilterSubject(expr);
	if (!subject) {
		return nullptr;
	}
	// the reference underneath identifies the column, in the same projection space the filter set uses
	vector<StructExtractPathEntry> path;
	auto &root = PeelStructExtractPath(*subject, path);
	if (root.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return nullptr;
	}
	auto &binding = root.Cast<BoundColumnRefExpression>().Binding();
	if (binding.table_index != info.table_index) {
		// a reference to a different table tells us nothing about this one
		// LCOV_EXCL_START
		return nullptr;
		// LCOV_EXCL_STOP
	}
	auto projection_index = binding.column_index;
	if (projection_index >= info.column_ids.size()) {
		// LCOV_EXCL_START
		return nullptr;
		// LCOV_EXCL_STOP
	}
	auto column_id = info.column_ids[projection_index];
	auto field_id = ResolveFilterField(*subject, column_id);
	if (!field_id) {
		return nullptr;
	}
	auto rewritten =
	    ExpressionIterator::ReplaceExpression(expr, *subject, BoundReferenceExpression(field_id->Type(), 0U));
	return make_uniq<DuckLakeFilterNode>(ColumnFilterInfo(field_id->GetFieldIndex().index, field_id->Type(),
	                                                      make_uniq<ExpressionFilter>(std::move(rewritten))));
}

void DuckLakeFileListData::AddFilterToPushdownInfo(FilterPushdownInfo &pushdown_info, column_t column_id,
                                                   unique_ptr<TableFilter> filter) const {
	auto node = GetFilterNode(column_id, std::move(filter));
	if (node) {
		AddFilterNodeToPushdownInfo(pushdown_info, *node);
	}
}

void DuckLakeFileListData::AddFilterNodeToPushdownInfo(FilterPushdownInfo &pushdown_info,
                                                       DuckLakeFilterNode &node) const {
	if (node.type == DuckLakeFilterNodeType::CONJUNCTION_AND) {
		// every conjunct has to hold, so each can be filtered on separately
		for (auto &child : node.children) {
			AddFilterNodeToPushdownInfo(pushdown_info, *child);
		}
		return;
	}
	if (node.type != DuckLakeFilterNodeType::COLUMN_FILTER) {
		// LCOV_EXCL_START
		return;
		// LCOV_EXCL_STOP
	}
	auto &column_filter = *node.column_filter;
	auto entry = pushdown_info.column_filters.find(column_filter.column_field_index);
	if (entry == pushdown_info.column_filters.end()) {
		pushdown_info.column_filters.emplace(column_filter.column_field_index, std::move(column_filter));
		return;
	}
	auto &existing_filter = entry->second.table_filter;
	existing_filter = make_uniq<ExpressionFilter>(DuckLakeUtil::MergeFilterExpressions(
	    std::move(existing_filter->expr), std::move(column_filter.table_filter->expr)));
}

unique_ptr<MultiFileList>
DuckLakeFileListData::DynamicFilterPushdown(MultiFileDynamicPushdownInfo &dynamic_pushdown_info) const {
	auto &options = dynamic_pushdown_info.options;
	auto &names = dynamic_pushdown_info.column_names;
	auto &types = dynamic_pushdown_info.column_types;
	auto &column_ids = dynamic_pushdown_info.column_ids;
	auto &context = dynamic_pushdown_info.context;
	auto &filters = dynamic_pushdown_info.filters;

	if (!SupportsFilterPushdown() || !filters.HasFilters()) {
		return nullptr;
	}

	// the filter set carries the static filters over, so the per-column filters are rebuilt from it - but a
	// TableFilterSet cannot round-trip a tree, so those have to be carried over by hand
	auto pushdown_info = make_uniq<FilterPushdownInfo>();
	if (filter_info) {
		for (const auto &tree : filter_info->filter_trees) {
			pushdown_info->filter_trees.push_back(tree.Copy());
		}
	}

	for (auto &entry : filters) {
		auto column_id = column_ids[entry.GetIndex().GetIndex()];
		AddFilterToPushdownInfo(
		    *pushdown_info, column_id,
		    ExpressionFilter::GetExpressionFilter(entry.Filter(), "DuckLakeMultiFileList::DynamicFilterPushdown")
		        .Copy());
	}

	if (pushdown_info->Empty()) {
		// no pushdown possible
		return nullptr;
	}

	return make_uniq<DuckLakeMultiFileList>(context, read_info, transaction_local_files, transaction_local_data,
	                                        std::move(pushdown_info));
}

template <class FUNC>
static void VisitFilterTreeLeaves(const DuckLakeFilterNode &node, FUNC &&visit) {
	if (node.type == DuckLakeFilterNodeType::COLUMN_FILTER) {
		visit(node);
		return;
	}
	for (const auto &child : node.children) {
		VisitFilterTreeLeaves(*child, visit);
	}
}

bool DuckLakeFileListData::FilterTreeState::VisitNode() {
	if (exhausted) {
		return false;
	}
	if (++visited_nodes > MAX_VISITED_NODES) {
		exhausted = true;
		return false;
	}
	return true;
}

bool DuckLakeFileListData::FilterTreeState::AddLeaves(const DuckLakeFilterNode &node) {
	if (exhausted) {
		// LCOV_EXCL_START
		return false;
		// LCOV_EXCL_STOP
	}
	VisitFilterTreeLeaves(node, [&](const DuckLakeFilterNode &) { leaves++; });
	if (leaves > MAX_LEAVES) {
		exhausted = true;
		return false;
	}
	return true;
}

//! Reduce an expression to per-column filters using the combiner, which also propagates equalities
unique_ptr<DuckLakeFilterNode> DuckLakeFileListData::CombineFilterNode(ClientContext &context,
                                                                       MultiFilePushdownInfo &info,
                                                                       const Expression &expr) const {
	FilterCombiner combiner(context);
	if (combiner.AddConjuncts(expr.Copy()) == FilterResult::UNSATISFIABLE) {
		return make_uniq<DuckLakeFilterNode>(DuckLakeFilterNodeType::MATCH_NONE);
	}
	vector<FilterPushdownResult> pushdown_results;
	auto table_filter_set = combiner.GenerateTableScanFilters(info.column_indexes, pushdown_results);
	if (combiner.HasFilters() || !table_filter_set.HasFilters()) {
		return nullptr;
	}
	auto result = make_uniq<DuckLakeFilterNode>(DuckLakeFilterNodeType::CONJUNCTION_AND);
	for (auto &entry : table_filter_set) {
		auto node = GetFilterNode(info.column_ids[entry.GetIndex().GetIndex()], entry.TakeFilter());
		if (node) {
			result->children.push_back(std::move(node));
		}
	}
	if (result->children.empty()) {
		return nullptr;
	}
	return std::move(result);
}

unique_ptr<DuckLakeFilterNode> DuckLakeFileListData::BuildFilterTree(ClientContext &context,
                                                                     MultiFilePushdownInfo &info,
                                                                     const Expression &expr,
                                                                     FilterTreeState &state) const {
	if (!state.VisitNode()) {
		return nullptr;
	}

	const bool is_or = expr.GetExpressionType() == ExpressionType::CONJUNCTION_OR;
	if (!is_or) {
		// the combiner sees all conjuncts at once, so let it try before splitting them up
		auto node = CombineFilterNode(context, info, expr);
		if (node) {
			return state.AddLeaves(*node) ? std::move(node) : nullptr;
		}
	}

	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
		auto &conjunction = expr.Cast<BoundConjunctionExpression>();
		auto result = make_uniq<DuckLakeFilterNode>(is_or ? DuckLakeFilterNodeType::CONJUNCTION_OR
		                                                  : DuckLakeFilterNodeType::CONJUNCTION_AND);
		bool complete = true;
		for (auto &child : conjunction.GetChildren()) {
			auto node = BuildFilterTree(context, info, *child, state);
			if (!node) {
				// a branch we cannot express prunes nothing, so the whole disjunction prunes nothing
				if (is_or) {
					complete = false;
					break;
				}
				continue;
			}
			if (node->type == DuckLakeFilterNodeType::MATCH_NONE) {
				// a branch that matches nothing drops out of a disjunction and decides a conjunction
				if (is_or) {
					continue;
				}
				return node;
			}
			result->children.push_back(std::move(node));
		}
		if (complete && result->children.empty() && is_or) {
			// every branch matched nothing
			return make_uniq<DuckLakeFilterNode>(DuckLakeFilterNodeType::MATCH_NONE);
		}
		if (complete && !result->children.empty()) {
			return std::move(result);
		}
	}

	if (state.exhausted) {
		// no point trying the fallbacks, they can only add leaves we have no budget for
		return nullptr;
	}
	if (is_or) {
		// the branches did not work out - the combiner may still express the disjunction on a single column
		auto node = CombineFilterNode(context, info, expr);
		if (node) {
			// LCOV_EXCL_START
			return state.AddLeaves(*node) ? std::move(node) : nullptr;
			// LCOV_EXCL_STOP
		}
	}
	// the combiner only expresses a subset of what we can evaluate against column stats
	auto node = GetExpressionFilterNode(info, expr);
	if (!node) {
		return nullptr;
	}
	return state.AddLeaves(*node) ? std::move(node) : nullptr;
}

unique_ptr<MultiFileList> DuckLakeFileListData::ComplexFilterPushdown(ClientContext &context,
                                                                      const MultiFileOptions &options,
                                                                      MultiFilePushdownInfo &info,
                                                                      vector<unique_ptr<Expression>> &filters) const {
	if (!SupportsFilterPushdown() || filters.empty()) {
		return nullptr;
	}

	FilterCombiner combiner(context);
	for (auto &filter : filters) {
		combiner.AddFilter(filter->Copy());
	}
	vector<FilterPushdownResult> pushdown_results;
	auto table_filter_set = combiner.GenerateTableScanFilters(info.column_indexes, pushdown_results);

	auto pushdown_info = filter_info ? filter_info->Copy() : make_uniq<FilterPushdownInfo>();

	for (auto &entry : table_filter_set) {
		auto column_id = info.column_ids[entry.GetIndex().GetIndex()];
		AddFilterToPushdownInfo(*pushdown_info, column_id, entry.TakeFilter());
	}

	// a disjunction cannot be reduced to per-column filters without losing the correlation between them
	for (auto &filter : filters) {
		if (filter->GetExpressionType() != ExpressionType::CONJUNCTION_OR) {
			continue;
		}
		bool already_pushed = false;
		for (auto &tree : pushdown_info->filter_trees) {
			if (tree.source->Equals(*filter)) {
				already_pushed = true;
				break;
			}
		}
		if (already_pushed) {
			continue;
		}
		FilterTreeState state;
		auto root = BuildFilterTree(context, info, *filter, state);
		if (!root || state.exhausted) {
			// too big to express as a tree - the per-column filters still apply
			continue;
		}
		unordered_set<idx_t> columns;
		VisitFilterTreeLeaves(
		    *root, [&](const DuckLakeFilterNode &leaf) { columns.insert(leaf.column_filter->column_field_index); });
		if (columns.size() < 2 && CombineFilterNode(context, info, *filter)) {
			// the combiner reduced the whole disjunction to its one column, so it is already pushed down
			// LCOV_EXCL_START
			continue;
			// LCOV_EXCL_STOP
		}
		DuckLakeFilterTree tree;
		tree.root = std::move(root);
		tree.source = filter->Copy();
		pushdown_info->filter_trees.push_back(std::move(tree));
	}

	if (pushdown_info->Empty()) {
		return nullptr;
	}

	return make_uniq<DuckLakeMultiFileList>(context, read_info, transaction_local_files, transaction_local_data,
	                                        std::move(pushdown_info));
}

idx_t DuckLakeFileListData::GetTotalFileCount() const {
	return GetFiles().size();
}

unique_ptr<NodeStatistics> DuckLakeFileListData::GetCardinality(ClientContext &context) const {
	auto stats = read_info.table.GetTableStats(context);
	if (!stats || stats->record_count_unknown) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(stats->record_count);
}

DuckLakeTableEntry &DuckLakeFileListData::GetTable() {
	return read_info.table;
}

//! The delete file as the value a table function that reads it takes - empty if there is no delete file
static string DeleteFileSQL(const DuckLakeFileData &delete_file) {
	if (delete_file.path.empty()) {
		return string();
	}
	OpenFileInfo delete_info(delete_file.path);
	delete_info.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
	auto &delete_options = delete_info.extended_info->options;
	delete_options["file_size"] = Value::UBIGINT(delete_file.file_size_bytes);
	if (delete_file.footer_size.IsValid()) {
		delete_options["footer_size"] = Value::UBIGINT(delete_file.footer_size.GetIndex());
	}
	if (!delete_file.encryption_key.empty()) {
		delete_options["encryption_key"] = Value::BLOB_RAW(delete_file.encryption_key);
	}
	if (delete_file.format == DeleteFileFormat::PUFFIN) {
		delete_options["format"] = Value("puffin");
	}
	delete_options["validate_external_file_cache"] = Value::BOOLEAN(false);
	delete_options["etag"] = Value("");
	delete_options["last_modified"] = Value::TIMESTAMP(timestamp_t(0));
	return delete_info.ToValue().ToSQLString();
}

//! The table function call that reads a delete file
static string ReadDeleteFileSQL(const DuckLakeFileData &delete_file) {
	return "ducklake_read_delete_file([" + DeleteFileSQL(delete_file) + "])";
}

void DuckLakeFileListData::AddDeleteScanOptions(const DuckLakeDeleteScanEntry &scan,
                                                ExtendedOpenFileInfo &extended_info) const {
	// the rows that were deleted in the snapshot range, with the snapshot that deleted them
	auto snapshot = to_string(scan.snapshot_id.GetIndex());
	string start_filter;
	if (scan.start_snapshot.IsValid()) {
		start_filter = " AND snapshot_id >= " + to_string(scan.start_snapshot.GetIndex());
	}
	if (scan.end_snapshot.IsValid()) {
		start_filter += " AND snapshot_id <= " + to_string(scan.end_snapshot.GetIndex());
	}
	if (scan.file_deleted && scan.delete_file.path.empty()) {
		// the file was deleted - all its rows, except for those that were deleted before or have a delete of their own
		string inlined_rows;
		string inlined_scanned;
		for (auto &inlined_delete : scan.inlined_file_deletions) {
			inlined_rows += (inlined_rows.empty() ? "" : ", ") + to_string(inlined_delete.first);
			if (inlined_delete.second >= scan.start_snapshot.GetIndex()) {
				inlined_scanned += (inlined_scanned.empty() ? "" : ", ") + string("(") +
				                   to_string(inlined_delete.first) + "::BIGINT, " + to_string(inlined_delete.second) +
				                   "::BIGINT)";
			}
		}
		auto query = StringUtil::Format("SELECT row_start, row_end, bits, %s::BIGINT AS row_value FROM "
		                                "ducklake_scan_rows([%s]",
		                                snapshot, DeleteFileSQL(scan.previous_delete_file));
		if (!inlined_rows.empty()) {
			query += ", deleted_rows := [" + inlined_rows + "]";
		}
		query += ")";
		if (!inlined_scanned.empty()) {
			query += " UNION ALL SELECT row_id, row_id + 1, NULL::BIT, row_value FROM (VALUES " + inlined_scanned +
			         ") AS inlined(row_id, row_value)";
		}
		extended_info.options["filter_query"] = Value(query);
		return;
	}
	string deleted_rows;
	if (!scan.delete_file.path.empty()) {
		// partial delete - the rows of the delete file, which can carry the snapshot of each of its deletes
		string insert_filter;
		if (scan.insert_snapshot.IsValid()) {
			// a row deleted in the snapshot that inserted it was never visible
			insert_filter =
			    " AND coalesce(snapshot_id, " + snapshot + ") <> " + to_string(scan.insert_snapshot.GetIndex());
		}
		auto out_of_range =
		    StringUtil::Format("Invalid delete data - delete index read from file %s is out of range for data file %s",
		                       scan.delete_file.path, scan.file.path);
		deleted_rows = "SELECT CASE WHEN pos >= " + to_string(scan.row_count) + " THEN error(" +
		               Value(out_of_range).ToSQLString() + ") ELSE pos END AS row_id, coalesce(snapshot_id, " +
		               snapshot + ") AS row_value FROM current_deletes WHERE (snapshot_id IS NULL OR (true" +
		               start_filter + "))" + insert_filter;
	} else {
		deleted_rows = "SELECT NULL::BIGINT AS row_id, NULL::BIGINT AS row_value WHERE false";
	}
	// inlined deletes replace the deletes of the same rows - those before the range are not part of it
	string query;
	if (!scan.delete_file.path.empty()) {
		query = "WITH current_deletes AS (SELECT pos, snapshot_id FROM " + ReadDeleteFileSQL(scan.delete_file) + "), ";
	} else {
		query = "WITH ";
	}
	query += "deleted_rows AS (" + deleted_rows + ")";
	string inlined_rows;
	string inlined_scanned;
	for (auto &inlined_delete : scan.inlined_file_deletions) {
		inlined_rows += (inlined_rows.empty() ? "" : ", ") + to_string(inlined_delete.first);
		if (inlined_delete.second >= scan.start_snapshot.GetIndex()) {
			inlined_scanned += (inlined_scanned.empty() ? "" : ", ") + string("(") + to_string(inlined_delete.first) +
			                   "::BIGINT, " + to_string(inlined_delete.second) + "::BIGINT)";
		}
	}
	query += ", all_deletes AS (SELECT row_id, row_value FROM deleted_rows";
	if (!inlined_rows.empty()) {
		query += " WHERE row_id NOT IN (" + inlined_rows + ")";
	}
	if (!inlined_scanned.empty()) {
		query +=
		    " UNION ALL SELECT row_id, row_value FROM (VALUES " + inlined_scanned + ") AS inlined(row_id, row_value)";
	}
	query += ")";
	query += " SELECT row_id, row_value FROM all_deletes";
	if (!scan.previous_delete_file.path.empty()) {
		// the deletes of the previous delete file are not new - unless the delete file has all deletes
		query += " WHERE " +
		         string(scan.delete_file.path.empty()
		                    ? "false"
		                    : "(SELECT coalesce(bool_or(snapshot_id IS NOT NULL), false) FROM current_deletes)") +
		         " OR row_id NOT IN (SELECT CASE WHEN pos >= " + to_string(scan.row_count) + " THEN error(" +
		         Value(StringUtil::Format("Invalid delete data - delete index read from file %s is out of range for "
		                                  "data file %s",
		                                  scan.previous_delete_file.path, scan.file.path))
		             .ToSQLString() +
		         ") ELSE pos END FROM " + ReadDeleteFileSQL(scan.previous_delete_file) + ")";
	}
	// every deleted row is a range of one row, with the snapshot that deleted it as its value
	extended_info.options["filter_query"] =
	    Value("SELECT row_id AS row_start, row_id + 1 AS row_end, NULL::BIT AS bits, row_value FROM (" + query + ")");
}

void DuckLakeFileListData::AddDeleteOptions(const DuckLakeFileListEntry &file_entry,
                                            ExtendedOpenFileInfo &extended_info) const {
	if (file_entry.delete_file.path.empty() && file_entry.inlined_file_deletions.empty()) {
		return;
	}
	// the snapshot up to which the deletes of the delete file apply
	optional_idx snapshot_filter;
	if (read_info.scan_type == DuckLakeScanType::SCAN_INSERTIONS) {
		// only the deletes of the snapshot that inserted the rows
		snapshot_filter = file_entry.snapshot_id.GetIndex();
	} else {
		auto transaction = read_info.GetTransaction();
		bool has_local_delete = read_info.include_local_changes && transaction &&
		                        transaction->HasLocalDeleteForFile(read_info.table_id, file_entry.file.path);
		if (!has_local_delete) {
			// only if this file is not part of a running transaction we are on the latest valid snapshot
			snapshot_filter = read_info.snapshot.snapshot_id;
		}
	}
	// inlined deletes apply in every snapshot
	vector<string> inlined_rows;
	for (auto &row : file_entry.inlined_file_deletions) {
		inlined_rows.push_back(to_string(row));
	}
	// the rows of the file that are not deleted
	auto query = StringUtil::Format("SELECT * FROM ducklake_scan_rows([%s]", DeleteFileSQL(file_entry.delete_file));
	if (snapshot_filter.IsValid()) {
		query += StringUtil::Format(", snapshot_filter := %d", snapshot_filter.GetIndex());
	}
	if (!inlined_rows.empty()) {
		query += StringUtil::Format(", deleted_rows := [%s]", StringUtil::Join(inlined_rows, ", "));
	}
	extended_info.options["filter_query"] = Value(query + ")");
}

OpenFileInfo DuckLakeFileListData::GetFile(idx_t i) const {
	auto &files = GetFiles();
	if (i >= files.size()) {
		return OpenFileInfo();
	}
	auto &file_entry = files[i];
	auto &file = file_entry.file;
	OpenFileInfo result(file.path);
	auto extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
	idx_t inlined_data_file_start = files.size() - inlined_data_tables.size();
	if (transaction_local_data) {
		inlined_data_file_start--;
	}
	if (transaction_local_data && i + 1 == files.size()) {
		// scanning transaction local data
		extended_info->options["transaction_local_data"] = Value::BOOLEAN(true);
		extended_info->options["inlined_data"] = Value::BOOLEAN(true);
		if (file_entry.row_id_start.IsValid()) {
			extended_info->options["row_id_start"] = Value::UBIGINT(file_entry.row_id_start.GetIndex());
		}
		extended_info->options["snapshot_id"] = Value(LogicalType::BIGINT);
		if (file_entry.mapping_id.IsValid()) {
			// LCOV_EXCL_START
			extended_info->options["mapping_id"] = Value::UBIGINT(file_entry.mapping_id.index);
			// LCOV_EXCL_STOP
		}
	} else if (i >= inlined_data_file_start) {
		// scanning inlined data
		auto inlined_data_index = i - inlined_data_file_start;
		auto &inlined_data_table = inlined_data_tables[inlined_data_index];
		extended_info->options["table_name"] = inlined_data_table.table_name;
		extended_info->options["inlined_data"] = Value::BOOLEAN(true);
		extended_info->options["schema_version"] =
		    Value::BIGINT(NumericCast<int64_t>(inlined_data_table.schema_version));
		if (file_entry.committed_rows_id.IsValid()) {
			extended_info->options["committed_rows_id"] = Value::UBIGINT(file_entry.committed_rows_id.GetIndex());
		}
	} else {
		extended_info->options["file_size"] = Value::UBIGINT(file.file_size_bytes);
		if (file.footer_size.IsValid()) {
			extended_info->options["footer_size"] = Value::UBIGINT(file.footer_size.GetIndex());
		}
		if (files[i].row_id_start.IsValid()) {
			extended_info->options["row_id_start"] = Value::UBIGINT(files[i].row_id_start.GetIndex());
		}
		Value snapshot_id;
		if (files[i].snapshot_id.IsValid()) {
			snapshot_id = Value::BIGINT(NumericCast<int64_t>(files[i].snapshot_id.GetIndex()));
		} else {
			snapshot_id = Value(LogicalType::BIGINT);
		}
		extended_info->options["snapshot_id"] = std::move(snapshot_id);
		if (!file.encryption_key.empty()) {
			extended_info->options["encryption_key"] = Value::BLOB_RAW(file.encryption_key);
		}
		// files managed by DuckLake are never modified - we can keep them cached
		extended_info->options["validate_external_file_cache"] = Value::BOOLEAN(false);
		// etag / last modified time can be set to dummy values
		extended_info->options["etag"] = Value("");
		extended_info->options["last_modified"] = Value::TIMESTAMP(timestamp_t(0));
		if (!file_entry.delete_file.path.empty()) {
			extended_info->options["has_deletes"] = Value::BOOLEAN(true);
		}
		if (read_info.scan_type != DuckLakeScanType::SCAN_DELETIONS) {
			AddDeleteOptions(file_entry, *extended_info);
		} else if (i < delete_scans.size()) {
			AddDeleteScanOptions(delete_scans[i], *extended_info);
		}
		if (file_entry.mapping_id.IsValid()) {
			extended_info->options["mapping_id"] = Value::UBIGINT(file_entry.mapping_id.index);
		}
		for (auto &entry : file_entry.column_min_max) {
			// a missing option is a statistic the file does not have
			auto &stats = entry.second;
			auto field_index = to_string(entry.first);
			if (stats.has_min) {
				extended_info->options["column_min." + field_index] = Value(stats.min);
			}
			if (stats.has_max) {
				extended_info->options["column_max." + field_index] = Value(stats.max);
			}
			if (stats.has_contains_nan) {
				extended_info->options["column_contains_nan." + field_index] = Value::BOOLEAN(stats.contains_nan);
			}
		}
		if (file_entry.snapshot_filter_max.IsValid()) {
			extended_info->options["snapshot_filter_max"] = Value::UBIGINT(file_entry.snapshot_filter_max.GetIndex());
		}
		if (file_entry.snapshot_filter_min.IsValid()) {
			extended_info->options["snapshot_filter_min"] = Value::UBIGINT(file_entry.snapshot_filter_min.GetIndex());
		}
		if (!file_entry.snapshot_filter_excluded.empty()) {
			vector<Value> excluded;
			for (auto snapshot_id : file_entry.snapshot_filter_excluded) {
				excluded.push_back(Value::UBIGINT(snapshot_id));
			}
			extended_info->options["snapshot_filter_excluded"] = Value::LIST(LogicalType::UBIGINT, std::move(excluded));
		}
	}
	result.extended_info = std::move(extended_info);
	return result;
}

unique_ptr<FunctionData> DuckLakeFileListData::Copy() const {
	unique_ptr<FilterPushdownInfo> filter_copy;
	if (filter_info) {
		filter_copy = filter_info->Copy();
	}
	auto result = make_uniq<DuckLakeFileListData>(read_info, transaction_local_files, transaction_local_data,
	                                              std::move(filter_copy));
	result->files = GetFiles();
	result->read_file_list = read_file_list;
	result->delete_scans = delete_scans;
	result->inlined_data_tables = inlined_data_tables;
	return std::move(result);
}

const DuckLakeFileListEntry &DuckLakeFileListData::GetFileEntry(idx_t file_idx) const {
	auto &files = GetFiles();
	return files[file_idx];
}

static DuckLakeFileData GetFileData(const DuckLakeDataFile &file) {
	DuckLakeFileData result;
	result.path = file.file_name;
	result.encryption_key = file.encryption_key;
	result.file_size_bytes = file.file_size_bytes;
	result.footer_size = file.footer_size;
	return result;
}

DuckLakeFileData DuckLakeFileListData::GetDeleteData(const DuckLakeDataFile &file) {
	if (file.delete_files.empty()) {
		return DuckLakeFileData();
	}
	return GetDeleteData(file.delete_files.back());
}

DuckLakeFileData DuckLakeFileListData::GetDeleteData(const DuckLakeDeleteFile &delete_file) {
	DuckLakeFileData result;
	result.path = delete_file.file_name;
	result.encryption_key = delete_file.encryption_key;
	result.file_size_bytes = delete_file.file_size_bytes;
	result.footer_size = delete_file.footer_size;
	result.format = delete_file.format;
	return result;
}

static void SetDeleteBeginSnapshot(DuckLakeFileListEntry &entry, optional_idx begin_snapshot) {
}

static void SetDeleteBeginSnapshot(DuckLakeFileListExtendedEntry &entry, optional_idx begin_snapshot) {
	entry.delete_file_begin_snapshot = begin_snapshot;
}

template <class ENTRY>
static void ApplyLocalFileChanges(DuckLakeTransaction &transaction, TableIndex table_id, vector<ENTRY> &entries) {
	if (transaction.HasDroppedFiles()) {
		entries.erase(std::remove_if(entries.begin(), entries.end(),
		                             [&](const ENTRY &entry) { return transaction.FileIsDropped(entry.file.path); }),
		              entries.end());
	}
	// if the transaction has any local deletes - apply them to the file list
	if (transaction.HasLocalDeletes(table_id)) {
		for (auto &file_entry : entries) {
			optional_idx begin_snapshot;
			if (transaction.GetLocalDeleteForFile(table_id, file_entry.file.path, file_entry.delete_file,
			                                      begin_snapshot)) {
				SetDeleteBeginSnapshot(file_entry, begin_snapshot);
			}
		}
	}
}

void DuckLakeFileListData::AddInlinedDataTables(DuckLakeTransaction &transaction) const {
	inlined_data_tables =
	    read_info.table.GetInlinedDataTables(transaction, read_info.snapshot, read_info.include_local_changes);
	for (auto &table : inlined_data_tables) {
		files.push_back(InlinedDataEntry(table));
	}
}

vector<DuckLakeFileListExtendedEntry> DuckLakeFileListData::GetFilesExtended() const {
	lock_guard<mutex> l(file_lock);
	vector<DuckLakeFileListExtendedEntry> result;
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	if (!IsTransactionLocal(read_info.table_id)) {
		// not a transaction local table - read the file list from the metadata store
		auto &metadata_manager = transaction.GetMetadataManager();
		result = metadata_manager.GetExtendedFilesForTable(read_info.table, read_info.snapshot, filter_info.get());
	}
	ApplyLocalFileChanges(transaction, read_info.table_id, result);
	idx_t transaction_row_start = DuckLakeConstants::TRANSACTION_LOCAL_ROW_ID_START;
	for (auto &file : transaction_local_files) {
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.row_count = file.row_count;
		file_entry.file = GetFileData(file);
		file_entry.delete_file = GetDeleteData(file);
		if (!file.delete_files.empty()) {
			file_entry.delete_file_begin_snapshot = file.delete_files.back().begin_snapshot;
		}
		file_entry.row_id_start = transaction_row_start;
		transaction_row_start += file.row_count;
		result.push_back(std::move(file_entry));
	}
	inlined_data_tables = read_info.table.GetInlinedDataTables(transaction, read_info.snapshot);
	for (auto &table : inlined_data_tables) {
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.file.path = table.table_name.GetIdentifierName();
		file_entry.row_count = 0;
		file_entry.row_id_start = 0;
		file_entry.data_type = DuckLakeDataType::INLINED_DATA;
		result.push_back(std::move(file_entry));
	}
	if (transaction_local_data) {
		// we have transaction local inlined data - create the dummy file entry
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.file.path = DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME;
		file_entry.row_count = transaction_local_data->data->Count();
		file_entry.row_id_start = transaction_row_start;
		file_entry.data_type = DuckLakeDataType::TRANSACTION_LOCAL_INLINED_DATA;
		result.push_back(std::move(file_entry));
	}
	return result;
}

void DuckLakeFileListData::ApplyLocalChanges(DuckLakeTransaction &transaction) const {
	ApplyLocalFileChanges(transaction, read_info.table_id, files);
	if (transaction.HasLocalInlinedFileDeletes(read_info.table_id)) {
		for (auto &file_entry : files) {
			if (file_entry.file_id.IsValid()) {
				transaction.GetLocalInlinedFileDeletesForFile(read_info.table_id, file_entry.file_id.index,
				                                              file_entry.inlined_file_deletions);
			}
		}
	}
}

void DuckLakeFileListData::SetScanColumns(const vector<MultiFileColumnDefinition> &global_columns,
                                          const vector<ColumnIndex> &column_ids) const {
	lock_guard<mutex> l(file_lock);
	scan_field_ids = make_uniq<unordered_set<int32_t>>();
	for (auto &column_id : column_ids) {
		auto index = column_id.GetPrimaryIndex();
		if (index == DuckLakeMultiFileReader::COLUMN_IDENTIFIER_SNAPSHOT_ID) {
			scan_field_ids->insert(MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID);
		}
		if (index >= global_columns.size()) {
			continue;
		}
		auto &identifier = global_columns[index].identifier;
		if (!identifier.IsNull() && identifier.type().id() == LogicalTypeId::INTEGER) {
			scan_field_ids->insert(IntegerValue::Get(identifier));
		}
	}
}

vector<DuckLakeFileListEntry> DuckLakeFileListData::ReadCommittedFiles(
    DuckLakeTransaction &transaction, identifier_map_t<shared_ptr<DuckLakeCommittedInlinedRows>> &inlined_rows) const {
	static constexpr idx_t MAX_ATTEMPTS = 10;
	auto &metadata_manager = transaction.GetMetadataManager();
	if (!metadata_manager.CommitsEachStatement() || inlined_data_tables.empty()) {
		return metadata_manager.GetFilesForTable(read_info.table, read_info.snapshot, filter_info.get(),
		                                         read_info.include_local_changes);
	}
	// every query sees the latest commits, so the inlined rows are read first and again if a file was written since
	for (idx_t attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
		optional_idx next_file_id;
		inlined_rows.clear();
		if (attempt > 0) {
			// the flush can have dropped an inlined table it emptied
			inlined_data_tables =
			    read_info.table.GetInlinedDataTables(transaction, read_info.snapshot, read_info.include_local_changes);
		}
		for (auto &table : inlined_data_tables) {
			vector<MultiFileColumnDefinition> columns;
			if (!DuckLakeMultiFileReader::GetInlinedTableColumns(read_info, table.schema_version, columns)) {
				continue;
			}
			if (scan_field_ids) {
				columns.erase(std::remove_if(columns.begin(), columns.end(),
				                             [&](const MultiFileColumnDefinition &col) {
					                             auto field_id = IntegerValue::Get(col.identifier);
					                             return field_id != MultiFileReader::ROW_ID_FIELD_ID &&
					                                    scan_field_ids->count(field_id) == 0;
				                             }),
				              columns.end());
			}
			inlined_rows[table.table_name] =
			    DuckLakeInlinedDataReader::ReadCommittedRows(read_info, table.table_name, columns, next_file_id);
		}
		auto result = metadata_manager.GetFilesForTable(read_info.table, read_info.snapshot, filter_info.get(),
		                                                read_info.include_local_changes);
		// without inlined rows a flush cannot move any of them
		bool written_since = next_file_id.IsValid() &&
		                     std::any_of(result.begin(), result.end(), [&](const DuckLakeFileListEntry &entry) {
			                     return entry.file_id.IsValid() && entry.file_id.index >= next_file_id.GetIndex();
		                     });
		if (!written_since) {
			return result;
		}
	}
	throw TransactionException("Transaction conflict - the inlined data of table %s kept moving to files while "
	                           "it was read",
	                           read_info.table.name);
}

void DuckLakeFileListData::GetFilesForTable() const {
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	inlined_data_tables =
	    read_info.table.GetInlinedDataTables(transaction, read_info.snapshot, read_info.include_local_changes);
	identifier_map_t<shared_ptr<DuckLakeCommittedInlinedRows>> inlined_rows;
	if (!IsTransactionLocal(read_info.table_id)) {
		files = ReadCommittedFiles(transaction, inlined_rows);
	}
	if (read_info.include_local_changes) {
		ApplyLocalChanges(transaction);
	}
	idx_t transaction_row_start = DuckLakeConstants::TRANSACTION_LOCAL_ROW_ID_START;
	for (auto &file : transaction_local_files) {
		DuckLakeFileListEntry file_entry;
		file_entry.file = GetFileData(file);
		file_entry.row_id_start = transaction_row_start;
		file_entry.delete_file = GetDeleteData(file);
		file_entry.mapping_id = file.mapping_id;
		transaction_row_start += file.row_count;
		files.emplace_back(std::move(file_entry));
	}
	for (auto &table : inlined_data_tables) {
		auto file_entry = InlinedDataEntry(table);
		auto rows = inlined_rows.find(table.table_name);
		if (rows != inlined_rows.end()) {
			file_entry.committed_rows_id = read_info.AddCommittedInlinedRows(rows->second);
		}
		files.push_back(std::move(file_entry));
	}
	if (transaction_local_data) {
		// we have transaction local inlined data - create the dummy file entry
		DuckLakeFileListEntry file_entry;
		file_entry.file.path = DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME;
		file_entry.row_id_start = transaction_row_start;
		file_entry.data_type = DuckLakeDataType::TRANSACTION_LOCAL_INLINED_DATA;
		files.push_back(std::move(file_entry));
	}
}

void DuckLakeFileListData::GetTableInsertions() const {
	if (IsTransactionLocal(read_info.table_id)) {
		throw InternalException("Cannot get changes between snapshots for transaction-local files");
	}
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	auto &metadata_manager = transaction.GetMetadataManager();
	files = metadata_manager.GetTableInsertions(read_info.table, *read_info.start_snapshot, read_info.snapshot);
	AddInlinedDataTables(transaction);
}

void DuckLakeFileListData::GetTableDeletions() const {
	if (IsTransactionLocal(read_info.table_id)) {
		throw InternalException("Cannot get changes between snapshots for transaction-local files");
	}
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	auto &metadata_manager = transaction.GetMetadataManager();
	delete_scans = metadata_manager.GetTableDeletions(read_info.table, *read_info.start_snapshot, read_info.snapshot,
	                                                  read_info.include_local_changes);
	for (auto &file : delete_scans) {
		DuckLakeFileListEntry file_entry;
		file_entry.file = file.file;
		file_entry.row_id_start = file.row_id_start;
		file_entry.snapshot_id = file.snapshot_id;
		file_entry.mapping_id = file.mapping_id;
		files.emplace_back(std::move(file_entry));
	}
	AddInlinedDataTables(transaction);
}

bool DuckLakeFileListData::CanUseGlobalStats() const {
	return read_info.CanUseGlobalStats();
}

bool DuckLakeFileListData::SupportsFilterPushdown() const {
	// a column retyped in this transaction keeps the stats written under its old type until commit
	return read_info.scan_type == DuckLakeScanType::SCAN_TABLE && !read_info.table.ChangedColumnType();
}

const vector<DuckLakeFileListEntry> &DuckLakeFileListData::GetFiles() const {
	lock_guard<mutex> l(file_lock);
	if (!read_file_list) {
		// we have not read the file list yet - read it
		switch (read_info.scan_type) {
		case DuckLakeScanType::SCAN_TABLE:
			GetFilesForTable();
			break;
		case DuckLakeScanType::SCAN_INSERTIONS:
			GetTableInsertions();
			break;
		case DuckLakeScanType::SCAN_DELETIONS:
			GetTableDeletions();
			break;
		default:
			throw InternalException("Unknown DuckLake scan type");
		}
		read_file_list = true;
	}
	return files;
}

//===--------------------------------------------------------------------===//
// File list function
//===--------------------------------------------------------------------===//
struct DuckLakeFileListGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static unique_ptr<GlobalTableFunctionState> DuckLakeFileListInit(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	return make_uniq<DuckLakeFileListGlobalState>();
}

static void DuckLakeFileListExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &data = input.bind_data->Cast<DuckLakeFileListData>();
	auto &state = input.global_state->Cast<DuckLakeFileListGlobalState>();
	auto total = data.GetTotalFileCount();
	idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - MinValue<idx_t>(state.offset, total));
	for (idx_t row = 0; row < count; row++) {
		auto file = data.GetFile(state.offset + row);
		output.SetValue(0, row, Value(file.path));
		vector<Value> keys;
		vector<Value> values;
		if (file.extended_info) {
			for (auto &option : file.extended_info->options) {
				keys.emplace_back(option.first);
				values.push_back(option.second.DefaultCastAs(LogicalType::VARIANT()));
			}
		}
		output.SetValue(1, row,
		                Value::MAP(LogicalType::VARCHAR, LogicalType::VARIANT(), std::move(keys), std::move(values)));
	}
	state.offset += count;
	output.SetChildCardinality(count);
}

static TableFunction DuckLakeFileListFunction() {
	TableFunction function("__ducklake_file_list", {}, DuckLakeFileListExecute, nullptr, DuckLakeFileListInit);
	return function;
}

static vector<LogicalType> DuckLakeFileListTypes() {
	return {LogicalType::VARCHAR, LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARIANT())};
}

DuckLakeMultiFileList::DuckLakeMultiFileList(ClientContext &context, unique_ptr<DuckLakeFileListData> data)
    : TableFunctionMultiFileList(context, DuckLakeFileListFunction(), std::move(data), DuckLakeFileListTypes()) {
}

DuckLakeMultiFileList::DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
                                             vector<DuckLakeDataFile> transaction_local_files,
                                             shared_ptr<DuckLakeInlinedData> transaction_local_data,
                                             unique_ptr<FilterPushdownInfo> filter_info)
    : DuckLakeMultiFileList(context, make_uniq<DuckLakeFileListData>(read_info, std::move(transaction_local_files),
                                                                     std::move(transaction_local_data),
                                                                     std::move(filter_info))) {
}

DuckLakeMultiFileList::DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
                                             vector<DuckLakeFileListEntry> files_to_scan)
    : DuckLakeMultiFileList(context, make_uniq<DuckLakeFileListData>(read_info, std::move(files_to_scan))) {
}

DuckLakeMultiFileList::DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
                                             const DuckLakeInlinedTableInfo &inlined_table)
    : DuckLakeMultiFileList(context, make_uniq<DuckLakeFileListData>(read_info, inlined_table)) {
}

unique_ptr<MultiFileList>
DuckLakeMultiFileList::DynamicFilterPushdown(MultiFileDynamicPushdownInfo &pushdown_info) const {
	return Data().DynamicFilterPushdown(pushdown_info);
}

unique_ptr<MultiFileList> DuckLakeMultiFileList::ComplexFilterPushdown(ClientContext &context,
                                                                       const MultiFileOptions &options,
                                                                       MultiFilePushdownInfo &info,
                                                                       vector<unique_ptr<Expression>> &filters) const {
	return Data().ComplexFilterPushdown(context, options, info, filters);
}

FileExpandResult DuckLakeMultiFileList::GetExpandResult() const {
	return FileExpandResult::MULTIPLE_FILES;
}

idx_t DuckLakeMultiFileList::GetTotalFileCount() const {
	return Data().GetTotalFileCount();
}

unique_ptr<MultiFileList> DuckLakeMultiFileList::Copy() const {
	auto data = unique_ptr_cast<FunctionData, DuckLakeFileListData>(Data().Copy());
	return unique_ptr<DuckLakeMultiFileList>(new DuckLakeMultiFileList(GetClientContext(), std::move(data)));
}

MultiFileCount DuckLakeMultiFileList::GetFileCount(idx_t min_exact_count) const {
	return MultiFileCount(GetTotalFileCount());
}

unique_ptr<NodeStatistics> DuckLakeMultiFileList::GetCardinality(ClientContext &context) const {
	return Data().GetCardinality(context);
}

} // namespace duckdb
