#include <ATen/VmapMode.h>
#include <ATen/Operators.h>
#include <torch/torch.h>

#include <functional>

namespace jxt
{
template <typename F>
class vmap
{
public:
  vmap(F && f, const std::vector<int64_t> & in_dims = {0}, const int64_t out_dim = 0)
    : _f(std::forward<F>(f)),
      _in_dims(in_dims),
      _out_dim(out_dim)
  {
  }

  template <typename... Args>
  auto operator()(Args &&... args)
  {
    // Get the current vmap level
    auto level = at::impl::VmapMode::increment_nesting();

    // Call the function on the batched input to get the batched output
    int i = 0;
    auto out_batched = _f(add_batch_dim(args, i++, level)...);

    // Remove the batch dimension from the batched output
    auto out = _remove_batch_dim(out_batched, level, _batch_size, _out_dim);

    // Finally, decrement the nesting level
    at::impl::VmapMode::decrement_nesting();

    return out;
  }

private:
  // Helper to extract the batch size and add the batch dimension to a tensor
  torch::Tensor add_batch_dim(const torch::Tensor & tensor, int64_t i, int64_t level)
  {
    _batch_size = tensor.sizes()[i];
    return i < 0 ? tensor : _add_batch_dim(tensor, _in_dims[i], level);
  }

  // Store this but let's hope that the compiler can optimize it away... who knows.
  F _f;

  const std::vector<int64_t> _in_dims;
  const int64_t _out_dim;
  int64_t _batch_size;
};
}
