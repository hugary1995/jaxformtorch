#include <torch/torch.h>

#include <functional>

namespace jxt
{
template <class F, class... Args>
std::result_of_t<F && (Args && ...)>
identity(F && f, Args &&... args)
{
  return f(args...);
}

template <typename Ret, typename... Args>
class vmap
{
public:
  vmap(const std::function<Ret(Args...)> & f)
    : _f(f)
  {
  }

  Ret operator()(Args &&... args) { return _f(args...); }

private:
  std::function<Ret(Args...)> _f;
};
}
