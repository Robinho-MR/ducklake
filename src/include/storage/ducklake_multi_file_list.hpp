//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_multi_file_list.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/table_function_multi_file_list.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_transaction.hpp"
#include "storage/ducklake_metadata_info.hpp"
#include "storage/ducklake_inlined_data.hpp"
#include "storage/ducklake_metadata_manager.hpp"

namespace duckdb {

class DuckLakeMultiFileList;

//! The bind data of the DuckLake file list function: the files of a DuckLake scan, and what they are scanned with
class DuckLakeFileListData : public TableFunctionData {
	static constexpr const char *DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME =
	    "__ducklake_inlined_transaction_local_data";

public:
	DuckLakeFileListData(DuckLakeFunctionInfo &read_info, vector<DuckLakeDataFile> transaction_local_files,
	                     shared_ptr<DuckLakeInlinedData> transaction_local_data,
	                     unique_ptr<FilterPushdownInfo> filter_info = nullptr);
	DuckLakeFileListData(DuckLakeFunctionInfo &read_info, vector<DuckLakeFileListEntry> files_to_scan);
	DuckLakeFileListData(DuckLakeFunctionInfo &read_info, const DuckLakeInlinedTableInfo &inlined_table);

	unique_ptr<MultiFileList> DynamicFilterPushdown(MultiFileDynamicPushdownInfo &pushdown_info) const;

	unique_ptr<MultiFileList> ComplexFilterPushdown(ClientContext &context, const MultiFileOptions &options,
	                                                MultiFilePushdownInfo &info,
	                                                vector<unique_ptr<Expression>> &filters) const;

	idx_t GetTotalFileCount() const;
	unique_ptr<NodeStatistics> GetCardinality(ClientContext &context) const;
	DuckLakeTableEntry &GetTable();
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override {
		return this == &other;
	}
	//! The i-th file, with the options to scan it with
	OpenFileInfo GetFile(idx_t i) const;
	vector<DuckLakeFileListExtendedEntry> GetFilesExtended() const;
	const vector<DuckLakeFileListEntry> &GetFiles() const;
	const DuckLakeFileListEntry &GetFileEntry(idx_t file_idx) const;
	//! Set the columns a scan reads, before its file list is read
	void SetScanColumns(const vector<MultiFileColumnDefinition> &global_columns,
	                    const vector<ColumnIndex> &column_ids) const;
	optional_ptr<const FilterPushdownInfo> GetFilterInfo() const {
		return filter_info.get();
	}

	bool CanUseGlobalStats() const;
	//! Whether filters can prune files by their stats
	bool SupportsFilterPushdown() const;
	static DuckLakeFileData GetDeleteData(const DuckLakeDataFile &file);
	static DuckLakeFileData GetDeleteData(const DuckLakeDeleteFile &delete_file);

private:
	//! The options that describe the deleted rows of a data file - read by the core scan
	void AddDeleteOptions(const DuckLakeFileListEntry &file_entry, ExtendedOpenFileInfo &extended_info) const;
	//! The options that describe the rows that were deleted from a data file - for scans of deletions
	void AddDeleteScanOptions(const DuckLakeDeleteScanEntry &scan, ExtendedOpenFileInfo &extended_info) const;
	void ApplyLocalChanges(DuckLakeTransaction &transaction) const;
	void GetFilesForTable() const;
	//! Read the committed files, and the committed inlined rows before them when each metadata query commits
	vector<DuckLakeFileListEntry>
	ReadCommittedFiles(DuckLakeTransaction &transaction,
	                   identifier_map_t<shared_ptr<DuckLakeCommittedInlinedRows>> &inlined_rows) const;
	void GetTableInsertions() const;
	void GetTableDeletions() const;
	void AddInlinedDataTables(DuckLakeTransaction &transaction) const;
	void AddFilterToPushdownInfo(FilterPushdownInfo &pushdown_info, column_t column_id,
	                             unique_ptr<TableFilter> filter) const;
	//! Build the node for a table filter on one column - a leaf, or a conjunction of them when it
	//! constrains several nested fields. nullptr for virtual columns and subjects that have no stats.
	unique_ptr<DuckLakeFilterNode> GetFilterNode(column_t column_id, unique_ptr<TableFilter> filter) const;
	//! Build the node for a filter expression reading one table column, possibly several of its fields
	unique_ptr<DuckLakeFilterNode> GetColumnFilterNode(column_t column_id, const Expression &expr,
	                                                   const LogicalType &column_type) const;
	//! Record the leaves of a node as single-column filters
	void AddFilterNodeToPushdownInfo(FilterPushdownInfo &pushdown_info, DuckLakeFilterNode &node) const;
	//! Build a leaf node directly from an expression that constrains a single column
	unique_ptr<DuckLakeFilterNode> GetExpressionFilterNode(MultiFilePushdownInfo &info, const Expression &expr) const;
	//! Resolve the field a filter subject reads, descending into nested fields
	optional_ptr<const DuckLakeFieldId> ResolveFilterField(const Expression &subject, column_t column_id) const;
	//! Reduce an expression to per-column filters using the FilterCombiner
	unique_ptr<DuckLakeFilterNode> CombineFilterNode(ClientContext &context, MultiFilePushdownInfo &info,
	                                                 const Expression &expr) const;
	//! Budgets for building one filter tree
	struct FilterTreeState {
		//! Bounds how much of the expression we walk - every step runs a FilterCombiner over it
		static constexpr idx_t MAX_VISITED_NODES = 64;
		//! Bounds what the generated query costs: one stats condition per leaf, all nested in one OR/AND
		//! chain. Separate from the walk, since one visit can yield a leaf per column. The leaves share
		//! the column's stats join, but planning the chain still grows superlinearly in its size.
		static constexpr idx_t MAX_LEAVES = 64;

		idx_t visited_nodes = 0;
		idx_t leaves = 0;
		//! Set when a budget runs out - the tree is then dropped rather than kept half-built
		bool exhausted = false;

		bool VisitNode();
		bool AddLeaves(const DuckLakeFilterNode &node);
	};
	//! Build a filter tree, or nullptr if nothing in the expression can be evaluated against column stats.
	//! A tree is not a promise that it prunes - that is decided when the SQL is generated.
	unique_ptr<DuckLakeFilterNode> BuildFilterTree(ClientContext &context, MultiFilePushdownInfo &info,
	                                               const Expression &expr, FilterTreeState &state) const;

private:
	mutable mutex file_lock;
	DuckLakeFunctionInfo &read_info;
	//! The set of files to read
	mutable vector<DuckLakeFileListEntry> files;
	mutable bool read_file_list;
	//! The set of transaction-local files
	vector<DuckLakeDataFile> transaction_local_files;
	//! Inlined transaction-local data
	shared_ptr<DuckLakeInlinedData> transaction_local_data;
	//! Inlined data tables
	mutable vector<DuckLakeInlinedTableInfo> inlined_data_tables;
	//! The field ids a scan reads, set before its file list is read, with the snapshot id column under its own id
	mutable unique_ptr<unordered_set<int32_t>> scan_field_ids;
	//! The set of delete scans, only used when scanning deleted tuples using ducklake_table_deletions
	mutable vector<DuckLakeDeleteScanEntry> delete_scans;
	//! Filter pushdown information
	unique_ptr<FilterPushdownInfo> filter_info;
};

//! The files of a DuckLake scan - a file list that runs the DuckLake file list function (over DuckLakeFileListData)
class DuckLakeMultiFileList : public TableFunctionMultiFileList {
public:
	DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
	                      vector<DuckLakeDataFile> transaction_local_files,
	                      shared_ptr<DuckLakeInlinedData> transaction_local_data,
	                      unique_ptr<FilterPushdownInfo> filter_info = nullptr);
	DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
	                      vector<DuckLakeFileListEntry> files_to_scan);
	DuckLakeMultiFileList(ClientContext &context, DuckLakeFunctionInfo &read_info,
	                      const DuckLakeInlinedTableInfo &inlined_table);

	unique_ptr<MultiFileList> DynamicFilterPushdown(MultiFileDynamicPushdownInfo &pushdown_info) const override;
	unique_ptr<MultiFileList> ComplexFilterPushdown(ClientContext &context, const MultiFileOptions &options,
	                                                MultiFilePushdownInfo &info,
	                                                vector<unique_ptr<Expression>> &filters) const override;
	FileExpandResult GetExpandResult() const override;
	idx_t GetTotalFileCount() const override;
	MultiFileCount GetFileCount(idx_t min_exact_count = 0) const override;
	unique_ptr<NodeStatistics> GetCardinality(ClientContext &context) const override;
	unique_ptr<MultiFileList> Copy() const override;

	DuckLakeTableEntry &GetTable() {
		return Data().GetTable();
	}
	vector<DuckLakeFileListExtendedEntry> GetFilesExtended() const {
		return Data().GetFilesExtended();
	}
	const vector<DuckLakeFileListEntry> &GetFiles() const {
		return Data().GetFiles();
	}
	const DuckLakeFileListEntry &GetFileEntry(idx_t file_idx) const {
		return Data().GetFileEntry(file_idx);
	}
	void SetScanColumns(const vector<MultiFileColumnDefinition> &global_columns,
	                    const vector<ColumnIndex> &column_ids) const {
		Data().SetScanColumns(global_columns, column_ids);
	}
	optional_ptr<const FilterPushdownInfo> GetFilterInfo() const {
		return Data().GetFilterInfo();
	}
	bool CanUseGlobalStats() const {
		return Data().CanUseGlobalStats();
	}
	bool SupportsFilterPushdown() const {
		return Data().SupportsFilterPushdown();
	}
	static DuckLakeFileData GetDeleteData(const DuckLakeDataFile &file) {
		return DuckLakeFileListData::GetDeleteData(file);
	}
	static DuckLakeFileData GetDeleteData(const DuckLakeDeleteFile &delete_file) {
		return DuckLakeFileListData::GetDeleteData(delete_file);
	}

protected:
	//! All files are known as soon as the metadata is read
	bool FileIsAvailable(idx_t i) const override {
		return true;
	}

private:
	DuckLakeMultiFileList(ClientContext &context, unique_ptr<DuckLakeFileListData> data);
	DuckLakeFileListData &Data() const {
		return GetBindData()->Cast<DuckLakeFileListData>();
	}
};

} // namespace duckdb
