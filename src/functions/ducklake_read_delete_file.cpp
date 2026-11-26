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

//! The scan rows function returns the rows of a data file that are not deleted, as the rows the filter query of a file
//! is made of: (row_start, row_end, bits). The deletes are those of the delete file (up to the snapshot filter) and
//! the listed rows. The rows are described per region of 65536 rows: a region without deletes is part of a range, a
//! region with few deletes is the ranges between them, and a region with many deletes is a bitmap
struct DuckLakeScanRowsData : public TableFunctionData {
	bool has_delete_file = false;
	DuckLakeFileData delete_file;
	optional_idx snapshot_filter;
	vector<idx_t> deleted_rows;
};

//! A piece of the rows to scan: the rows from start up to but not including end (the end of the file if end is not
//! set), or - for a window - the rows of the region that start at start, except for the deleted rows of the window
struct DuckLakeScanPiece {
	idx_t start;
	optional_idx end;
	bool is_window = false;
	//! The range of the sorted deleted rows that are in the window
	idx_t deleted_begin = 0;
	idx_t deleted_end = 0;
};

struct DuckLakeScanRowsState : public GlobalTableFunctionState {
	vector<idx_t> deleted_rows;
	vector<DuckLakeScanPiece> pieces;
	idx_t piece_idx = 0;
};

//! The number of rows of a region, and the number of deletes in it from which it is a bitmap
static constexpr idx_t SCAN_REGION_SIZE = 65536;
static constexpr idx_t SCAN_REGION_BITMAP_THRESHOLD = 512;

static unique_ptr<FunctionData> DuckLakeScanRowsBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto result = make_uniq<DuckLakeScanRowsData>();
	auto &files = input.inputs[0];
	if (!files.IsNull()) {
		auto &children = ListValue::GetChildren(files);
		if (children.size() > 1) {
			throw InvalidInputException("ducklake_scan_rows reads at most one delete file");
		}
		if (!children.empty()) {
			result->has_delete_file = true;
			result->delete_file = ParseDeleteFile(children[0]);
		}
	}
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
	names.emplace_back("row_start");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("row_end");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("bits");
	return_types.emplace_back(LogicalType::BIT);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DuckLakeScanRowsInit(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<DuckLakeScanRowsData>();
	auto result = make_uniq<DuckLakeScanRowsState>();
	auto &deleted = result->deleted_rows;
	if (data.has_delete_file) {
		auto scan =
		    DuckLakeDeleteFilter::ScanDeleteFile(context, data.delete_file, optional_idx(), data.snapshot_filter);
		deleted = std::move(scan.deleted_rows);
	}
	deleted.insert(deleted.end(), data.deleted_rows.begin(), data.deleted_rows.end());
	if (!std::is_sorted(deleted.begin(), deleted.end())) {
		sort(deleted.begin(), deleted.end());
	}
	deleted.erase(std::unique(deleted.begin(), deleted.end()), deleted.end());

	auto &pieces = result->pieces;
	idx_t next_row = 0;
	for (idx_t i = 0; i < deleted.size();) {
		auto region_start = deleted[i] / SCAN_REGION_SIZE * SCAN_REGION_SIZE;
		auto region_end = region_start + SCAN_REGION_SIZE;
		idx_t end_idx = i;
		while (end_idx < deleted.size() && deleted[end_idx] < region_end) {
			end_idx++;
		}
		if (region_start > next_row) {
			// the regions in between do not have deletes
			pieces.push_back(DuckLakeScanPiece {next_row, optional_idx(region_start)});
		}
		if (end_idx - i >= SCAN_REGION_BITMAP_THRESHOLD) {
			pieces.push_back(DuckLakeScanPiece {region_start, optional_idx(), true, i, end_idx});
		} else {
			// the rows between the deleted rows
			auto row = region_start;
			for (auto delete_idx = i; delete_idx < end_idx; delete_idx++) {
				if (deleted[delete_idx] > row) {
					pieces.push_back(DuckLakeScanPiece {row, optional_idx(deleted[delete_idx])});
				}
				row = deleted[delete_idx] + 1;
			}
			if (row < region_end) {
				pieces.push_back(DuckLakeScanPiece {row, optional_idx(region_end)});
			}
		}
		next_row = region_end;
		i = end_idx;
	}
	// the rest of the file
	pieces.push_back(DuckLakeScanPiece {next_row, optional_idx()});
	return std::move(result);
}

static void DuckLakeScanRowsExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<DuckLakeScanRowsState>();
	auto starts = FlatVector::GetDataMutable<int64_t>(output.data[0]);
	auto ends = FlatVector::GetDataMutable<int64_t>(output.data[1]);
	auto &end_validity = FlatVector::ValidityMutable(output.data[1]);
	auto bitmaps = FlatVector::GetDataMutable<string_t>(output.data[2]);
	auto &bitmap_validity = FlatVector::ValidityMutable(output.data[2]);
	idx_t count = 0;
	while (count < STANDARD_VECTOR_SIZE && state.piece_idx < state.pieces.size()) {
		auto &piece = state.pieces[state.piece_idx++];
		starts[count] = NumericCast<int64_t>(piece.start);
		if (piece.is_window || !piece.end.IsValid()) {
			end_validity.SetInvalid(count);
		} else {
			ends[count] = NumericCast<int64_t>(piece.end.GetIndex());
		}
		if (piece.is_window) {
			// all rows of the region except for the deleted rows
			auto bitmap = StringVector::EmptyString(output.data[2], Bit::ComputeBitstringLen(SCAN_REGION_SIZE));
			Bit::SetEmptyBitString(bitmap, SCAN_REGION_SIZE);
			// the region has no padding bits: the first row is the highest bit of the first byte after the padding byte
			auto data = reinterpret_cast<uint8_t *>(bitmap.GetDataWriteable());
			memset(data + 1, 0xFF, bitmap.GetSize() - 1);
			for (auto delete_idx = piece.deleted_begin; delete_idx < piece.deleted_end; delete_idx++) {
				auto row = state.deleted_rows[delete_idx] - piece.start;
				data[1 + row / 8] = static_cast<uint8_t>(data[1 + row / 8] & ~(1 << (7 - row % 8)));
			}
			bitmap.Finalize();
			bitmaps[count] = bitmap;
		} else {
			bitmap_validity.SetInvalid(count);
		}
		count++;
	}
	output.SetChildCardinality(count);
}

TableFunctionSet DuckLakeScanRowsFunction::GetFunctions() {
	TableFunctionSet set("ducklake_scan_rows");
	TableFunction function("ducklake_scan_rows", {LogicalType::LIST(LogicalType::ANY)}, DuckLakeScanRowsExecute,
	                       DuckLakeScanRowsBind, DuckLakeScanRowsInit);
	function.GetSignature().WithTypedKwargs("options", [&](TypedKwargs &options) {
		options.Add("snapshot_filter", LogicalType::UBIGINT).Add("deleted_rows", LogicalType::LIST(LogicalType::UBIGINT));
	});
	set.AddFunction(std::move(function));
	return set;
}

} // namespace duckdb
