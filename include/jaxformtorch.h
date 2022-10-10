#include <ATen/VmapMode.h>
#include <ATen/Operators.h>
#include <torch/torch.h>

namespace jxt
{
template <typename F>
class vmap
{
public:
  vmap(F & f, const std::vector<int64_t> & in_dims = {0}, const int64_t out_dim = 0)
    : _f(std::forward<F>(f)),
      _in_dims(in_dims),
      _out_dim(out_dim)
  {
  }

  vmap(F && f, const std::vector<int64_t> & in_dims = {0}, const int64_t out_dim = 0)
    : _f(std::forward<F>(f)),
      _in_dims(in_dims),
      _out_dim(out_dim)
  {
  }

  template <typename... Args>
  auto operator()(Args &&... args)
  {
    // Dummy counter in fold expressions
    int i = 0;

    // Get the batch sizes
    std::vector<int64_t> batch_sizes;
    (batch_sizes.push_back(args.sizes()[_in_dims[i++]]), ...);
    int64_t batch_size = get_batch_size(batch_sizes);

    // Get the current vmap level
    int64_t level = at::impl::VmapMode::increment_nesting();

    // Call the function on the batched input to get the batched output
    i = 0;
    auto out_batched = _f(add_batch_dim(args, _in_dims[i++], level)...);

    // Remove the batch dimension from the batched output
    auto out = _remove_batch_dim(out_batched, level, batch_size, _out_dim);

    // Finally, decrement the nesting level
    at::impl::VmapMode::decrement_nesting();

    return out;
  }

private:
  // Get and validate the batch size
  int64_t get_batch_size(const std::vector<int64_t> & batch_sizes) const
  {
    int64_t batch_size = batch_sizes[0];
    const bool valid = std::all_of(batch_sizes.begin() + 1,
                                   batch_sizes.end(),
                                   [&](const int64_t & r) { return r == batch_size; });
    if (!valid)
      std::runtime_error("Batch sizes of arguments of a vmapped lambda are not consistent.");
    return batch_size;
  }

  // Add batch dimension
  torch::Tensor add_batch_dim(const torch::Tensor & tensor, int64_t dim, int64_t level) const
  {
    // If dim is negative then don't batch it
    return dim < 0 ? tensor : _add_batch_dim(tensor, dim, level);
  }

  // Store this but let's hope that the compiler can optimize it away... who knows.
  F _f;

  const std::vector<int64_t> _in_dims;
  const int64_t _out_dim;
};

template <typename F>
class vjp
{
public:
  vjp(F & f, const torch::Tensor & arg)
    : _arg(arg)
  {
    _arg.set_requires_grad(true);
    _out = f(_arg);
  }

  vjp(F && f, const torch::Tensor & arg)
    : _arg(arg)
  {
    _arg.set_requires_grad(true);
    _out = f(_arg);
  }

  const torch::Tensor & value() const { return _out; }

  torch::Tensor operator()(const torch::Tensor & cotangent)
  {
    return torch::autograd::grad({_out}, {_arg}, {cotangent}, true, true)[0];
  }

private:
  torch::Tensor _arg;
  torch::Tensor _out;
};

inline torch::Tensor
jacrev(const torch::Tensor & out, const torch::Tensor & arg)
{
  const torch::Tensor cotangents = torch::eye(out.sizes()[0]);
  torch::Tensor jac =
      vmap([&](const torch::Tensor & cotangent)
           { return torch::autograd::grad({out}, {arg}, {cotangent}, true, true)[0]; })(cotangents);
  return jac;
}
}
