#include "module/module.h"

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "tensor_impl.h"

namespace dllm::module {
void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  for (const auto& child : children_) {
    auto qualified_name = fmt::format("{}.{}", name_prefix, child.key());
    function(qualified_name, child.value());
    child.value()->apply_to_submodules(function, qualified_name);
  }
}

std::shared_ptr<Module> Module::shared_from_this_checked() const {
  std::shared_ptr<const Module> ptr;
  try {
    ptr = shared_from_this();
  } catch (const std::bad_weak_ptr&) {
    DLLM_ASSERT_TRUE(
        false,
        "It looks like you attempted to retrieve your top-level module "
        "as a shared_ptr, but it is not stored in a shared_ptr. "
        "Use std::make_shared instead of creating your module on "
        "the stack, or alternatively do not try to access your top-level "
        "module at all by passing /*include_self=*/false "
        "to modules() or named_modules()");
  }
  return std::const_pointer_cast<Module>(ptr);
}

void Module::apply(const ConstNamedModuleApplyFunction& function,
                   const std::string& name_prefix) const {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](const std::string& name,
                  const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      name_prefix);
}

OrderedDict<std::string, std::shared_ptr<State>> Module::named_states(
    const bool recurse) const {
  OrderedDict<std::string, std::shared_ptr<State>> result;
  if (!recurse) {
    for (const auto& state : states_) {
      result.insert(state.key(), state.value());
    }
  } else {
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& state : module.named_states(/*recurse=*/false)) {
        result.insert(fmt::format("{}.{}", name, state.key()), state.value());
      }
    });
  }
  return result;
}

OrderedDict<std::string, Tensor> Module::named_parameters(
    const bool recurse) const {
  auto states = named_states(recurse);
  OrderedDict<std::string, Tensor> result{};
  for (const auto& pairState : states) {
    for (auto parameters = pairState.value()->parameters();
         const auto& pairTensor : parameters) {
      result.insert(fmt::format("{}.{}", pairState.key(), pairTensor.key()),
                    pairTensor.value());
    }
  }
  return result;
}

void Module::to(TensorOptions options) const {
  auto parameters = named_parameters();
  for (const auto& pair : parameters) {
    pair.value().impl()->tensor() = pair.value().impl()->tensor().to(options);
  }
}

void Module::register_state(std::string name, std::shared_ptr<State> state) {
  states_.insert(std::move(name), std::move(state));
}
}  // namespace dllm::module

void dllm::save(const module::Module& module, const std::string& path) {
  auto dict = module.named_parameters();
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  // 创建字段
  std::shared_ptr<arrow::Field> string_field =
      arrow::field("key", arrow::utf8());
  std::shared_ptr<arrow::Field> vector_field = arrow::field(
      "value", arrow::list(arrow::field("item", arrow::float32())));
  auto fields = {string_field, vector_field};
  auto schema = std::make_shared<arrow::Schema>(fields);

  // 构建列数据
  arrow::StringBuilder keys_builder(pool);
  arrow::ListBuilder values_builder(
      pool, std::make_shared<arrow::FloatBuilder>(pool));
  auto* float_builder =
      dynamic_cast<arrow::FloatBuilder*>(values_builder.value_builder());

  for (const auto& pair : dict) {
    DLLM_ASSERT_TRUE(keys_builder.Append(pair.key()).ok(), "Writing failed");
    auto tensor = pair.value().impl()->tensor().cpu().to(at::kFloat);
    auto tensor_view = tensor.view({-1});
    DLLM_ASSERT_TRUE(values_builder.Append().ok(), "Writing failed");
    for (int64_t i = 0; i < tensor_view.numel(); ++i) {
      DLLM_ASSERT_TRUE(
          float_builder->Append(tensor_view[i].item().toFloat()).ok(),
          "Writing failed");
    }
  }

  // 完成构建
  std::shared_ptr<arrow::Array> keys_array, values_array;
  DLLM_ASSERT_TRUE(keys_builder.Finish(&keys_array).ok(), "Writing failed");
  DLLM_ASSERT_TRUE(values_builder.Finish(&values_array).ok(), "Writing failed");

  // 创建 Arrow 表
  std::shared_ptr<arrow::Table> table =
      arrow::Table::Make(schema, {keys_array, values_array});

  // 写入 Parquet 文件
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(path));
  PARQUET_THROW_NOT_OK(
      parquet::arrow::WriteTable(*table, pool, outfile, dict.size()));
}

void dllm::load(const module::Module& module, const std::string& path) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  // 打开文件
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(path, pool));

  // 读取 Parquet 文件到 Arrow 表
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto statusOpenFile = parquet::arrow::OpenFile(infile, pool, &reader);
  DLLM_ASSERT_TRUE(statusOpenFile.ok(), "Open File failed with error {}",
                   statusOpenFile.ToString());

  std::shared_ptr<arrow::Table> table;
  auto statusReadTable = reader->ReadTable(&table);
  DLLM_ASSERT_TRUE(statusReadTable.ok(), "Read File failed with error {}",
                   statusReadTable.ToString());

  // 解析 Arrow 表
  const std::shared_ptr<arrow::StringArray> keys_array =
      std::dynamic_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
  const std::shared_ptr<arrow::ListArray> values_array =
      std::dynamic_pointer_cast<arrow::ListArray>(table->column(1)->chunk(0));
  const std::shared_ptr<arrow::FloatArray> float_array =
      std::dynamic_pointer_cast<arrow::FloatArray>(values_array->values());

  auto parameter = module.named_parameters();

  // 重建模型参数
  for (int64_t i = 0; i < keys_array->length(); ++i) {
    std::string key = keys_array->GetString(i);
    const auto subarray_offset = values_array->value_offset(i);
    const auto subarray_length = values_array->value_length(i);
    std::vector<float> values;
    for (int64_t j = 0; j < subarray_length; ++j) {
      values.push_back(float_array->Value(subarray_offset + j));
    }
    at::Tensor tensor = torch::tensor(values, torch::dtype(at::kFloat));
    const auto find = parameter.find(key);
    DLLM_ASSERT_TRUE(find != nullptr, "wrong parameter name {} in saved model",
                     key);
    (void)find->impl()->tensor().copy_(
        tensor.to(find->options()).view(find->sizes()));
  }
}
