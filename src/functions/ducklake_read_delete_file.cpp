#include "functions/ducklake_table_functions.hpp"
#include "storage/ducklake_delete_filter.hpp"
#include "storage/ducklake_metadata_info.hpp"

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/types/bit.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

//! The delete function reads a DuckLake delete file - a parquet positional delete file or a puffin deletion vector -
//! and returns the deleted positions and the snapshots that deleted them. A scan runs it in the "filter_query" of a
//! data file whose rows were deleted
struct DuckLakeReadDeleteFileData : public TableFunctionData {
	DuckLakeFileData delete_file;
};

struct DuckLakeReadDeleteFileState : public GlobalTableFunctionState {
	bool scanned = false;
	DeleteFileScanResult result;
	idx_t offset = 0;
};

static DuckLakeFileData ParseDeleteFile(const Value &file_value) {
	auto file = OpenFileInfo::FromValue(file_value, Identifier("ducklake_read_delete_file"));
	DuckLakeFileData result;
	result.path = file.path;
	if (file.extended_info) {
		auto &info = *file.extended_info;
		idx_t file_size = 0;
		if (info.TryGetOption("file_size", file_size)) {
			result.file_size_bytes = file_size;
		}
		idx_t footer_size = 0;
		if (info.TryGetOption("footer_size", footer_size)) {
			result.footer_size = footer_size;
		}
		string encryption_key;
		if (info.TryGetOption("encryption_key", encryption_key)) {
			result.encryption_key = std::move(encryption_key);
		}
		string format;
		if (info.TryGetOption("format", format) && format == "puffin") {
			result.format = DeleteFileFormat::PUFFIN;
		}
	}
	return result;
}

static unique_ptr<FunctionData> DuckLakeReadDeleteFileBind(ClientContext &context, TableFunctionBindInput &input,
                                                           vector<LogicalType> &return_types,
                                                           vector<Identifier> &names) {
	auto &files = input.inputs[0];
	if (files.IsNull() || ListValue::GetChildren(files).size() != 1) {
		throw InvalidInputException("ducklake_read_delete_file reads exactly one delete file");
	}
	auto result = make_uniq<DuckLakeReadDeleteFileData>();
	result->delete_file = ParseDeleteFile(ListValue::GetChildren(files)[0]);
	names.emplace_back("pos");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("snapshot_id");
	return_types.emplace_back(LogicalType::BIGINT);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DuckLakeReadDeleteFileInit(ClientContext &context,
                                                                       TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<DuckLakeReadDeleteFileData>();
	auto result = make_uniq<DuckLakeReadDeleteFileState>();
	result->result = DuckLakeDeleteFilter::ScanDeleteFile(context, data.delete_file, optional_idx(), optional_idx());
	return std::move(result);
}

static void DuckLakeReadDeleteFileExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<DuckLakeReadDeleteFileState>();
	auto &rows = state.result.deleted_rows;
	auto &snapshots = state.result.snapshot_ids;
	idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, rows.size() - state.offset);
	auto pos_data = FlatVector::GetDataMutable<int64_t>(output.data[0]);
	auto snapshot_data = FlatVector::GetDataMutable<int64_t>(output.data[1]);
	auto &snapshot_validity = FlatVector::ValidityMutable(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		pos_data[i] = NumericCast<int64_t>(rows[state.offset + i]);
		if (state.offset + i < snapshots.size()) {
			snapshot_data[i] = NumericCast<int64_t>(snapshots[state.offset + i]);
		} else {
			snapshot_validity.SetInvalid(i);
		}
	}
	state.offset += count;
	output.SetChildCardinality(count);
}

TableFunctionSet DuckLakeReadDeleteFileFunction::GetFunctions() {
	TableFunctionSet set("ducklake_read_delete_file");
	TableFunction function("ducklake_read_delete_file", {LogicalType::LIST(LogicalType::ANY)},
	                       DuckLakeReadDeleteFileExecute, DuckLakeReadDeleteFileBind, DuckLakeReadDeleteFileInit);
	set.AddFunction(std::move(function));
	return set;
}

//! The bitmap function returns the rows that are deleted from a data file as a single BIT value - bit i is the row with
//! id i. The rows are those of the delete file (the deletions up to the snapshot filter) and the listed rows
struct DuckLakeReadDeleteBitmapData : public TableFunctionData {
	bool has_delete_file = false;
	DuckLakeFileData delete_file;
	idx_t row_count = 0;
	optional_idx snapshot_filter;
	vector<idx_t> deleted_rows;
};

struct DuckLakeReadDeleteBitmapState : public GlobalTableFunctionState {
	bool done = false;
};

static unique_ptr<FunctionData> DuckLakeReadDeleteBitmapBind(ClientContext &context, TableFunctionBindInput &input,
                                                             vector<LogicalType> &return_types,
                                                             vector<Identifier> &names) {
	auto result = make_uniq<DuckLakeReadDeleteBitmapData>();
	auto &files = input.inputs[0];
	if (!files.IsNull()) {
		auto &children = ListValue::GetChildren(files);
		if (children.size() > 1) {
			throw InvalidInputException("ducklake_read_delete_bitmap reads at most one delete file");
		}
		if (!children.empty()) {
			result->has_delete_file = true;
			result->delete_file = ParseDeleteFile(children[0]);
		}
	}
	auto row_count = input.named_parameters.find("row_count");
	if (row_count == input.named_parameters.end() || row_count->second.IsNull()) {
		throw InvalidInputException("ducklake_read_delete_bitmap needs the row_count of the data file");
	}
	result->row_count = UBigIntValue::Get(row_count->second.DefaultCastAs(LogicalType::UBIGINT));
	auto snapshot_filter = input.named_parameters.find("snapshot_filter");
	if (snapshot_filter != input.named_parameters.end() && !snapshot_filter->second.IsNull()) {
		result->snapshot_filter = UBigIntValue::Get(snapshot_filter->second.DefaultCastAs(LogicalType::UBIGINT));
	}
	auto deleted_rows = input.named_parameters.find("deleted_rows");
	if (deleted_rows != input.named_parameters.end() && !deleted_rows->second.IsNull()) {
		for (auto &row : ListValue::GetChildren(deleted_rows->second)) {
			result->deleted_rows.push_back(UBigIntValue::Get(row.DefaultCastAs(LogicalType::UBIGINT)));
		}
	}
	names.emplace_back("deleted");
	return_types.emplace_back(LogicalType::BIT);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DuckLakeReadDeleteBitmapInit(ClientContext &context,
                                                                         TableFunctionInitInput &input) {
	return make_uniq<DuckLakeReadDeleteBitmapState>();
}

static void DuckLakeReadDeleteBitmapExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<DuckLakeReadDeleteBitmapState>();
	auto &data = input.bind_data->Cast<DuckLakeReadDeleteBitmapData>();
	if (state.done) {
		output.SetChildCardinality(0);
		return;
	}
	state.done = true;
	auto bitmap = StringVector::EmptyString(output.data[0], Bit::ComputeBitstringLen(data.row_count));
	Bit::SetEmptyBitString(bitmap, data.row_count);
	auto mark_deleted = [&](const vector<idx_t> &rows) {
		for (auto row : rows) {
			if (row < data.row_count) {
				Bit::SetBit(bitmap, row, 1);
			}
		}
	};
	if (data.has_delete_file) {
		auto scan =
		    DuckLakeDeleteFilter::ScanDeleteFile(context, data.delete_file, optional_idx(), data.snapshot_filter);
		mark_deleted(scan.deleted_rows);
	}
	mark_deleted(data.deleted_rows);
	bitmap.Finalize();
	FlatVector::GetDataMutable<string_t>(output.data[0])[0] = bitmap;
	output.SetChildCardinality(1);
}

TableFunctionSet DuckLakeReadDeleteBitmapFunction::GetFunctions() {
	TableFunctionSet set("ducklake_read_delete_bitmap");
	TableFunction function("ducklake_read_delete_bitmap", {LogicalType::LIST(LogicalType::ANY)},
	                       DuckLakeReadDeleteBitmapExecute, DuckLakeReadDeleteBitmapBind,
	                       DuckLakeReadDeleteBitmapInit);
	function.GetSignature().WithTypedKwargs("options", [&](TypedKwargs &options) {
		options.Add("row_count", LogicalType::UBIGINT)
		    .Add("snapshot_filter", LogicalType::UBIGINT)
		    .Add("deleted_rows", LogicalType::LIST(LogicalType::UBIGINT));
	});
	set.AddFunction(std::move(function));
	return set;
}

} // namespace duckdb
