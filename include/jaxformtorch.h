#include <torch/torch.h>

#include <functional>

namespace jxt
{
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
  // Store this but let's hope that the compiler can optimize it away... who knows.
  std::function<Ret(Args...)> _f;
};
}
