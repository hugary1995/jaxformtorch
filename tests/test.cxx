#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <cmath>
#include "jaxformtorch.h"

TEST_CASE("vmap a lambda", "[function]")
{
  int64_t batch_size = 5;
  int64_t n = 3;
  torch::Tensor x = torch::randn({batch_size, n});

  // vmapped norm
  torch::Tensor norm_x = jxt::vmap([](const torch::Tensor & x) { return torch::norm(x); })(x);
  // 2-norm along axis 1
  torch::Tensor correct = torch::norm(x, 2, 1);

  REQUIRE(torch::allclose(norm_x, correct));
}
