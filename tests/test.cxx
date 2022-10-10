#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <cmath>
#include "jaxformtorch.h"

TEST_CASE("vmap a lambda", "[vmap]")
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

TEST_CASE("vmap matmul", "[vmap]")
{
  int64_t batch_size = 5;
  torch::Tensor x = torch::randn({batch_size, 3});
  torch::Tensor C = torch::randn({batch_size, 2, 3});

  torch::Tensor y = jxt::vmap([](const torch::Tensor & C, const torch::Tensor & x)
                              { return torch::matmul(C, x); },
                              {0, 0})(C, x);

  REQUIRE(torch::allclose(y, torch::matmul(C, x.unsqueeze(2)).squeeze(2)));
}

TEST_CASE("vjp", "[vjp]")
{
  torch::Tensor x = torch::randn({3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor cotangents = torch::eye(2);

  auto f = [&](const torch::Tensor & x) { return torch::matmul(C, x); };

  torch::Tensor jac_row1 = jxt::vjp(f, x)(cotangents.index({0}));
  torch::Tensor jac_row2 = jxt::vjp(f, x)(cotangents.index({1}));

  REQUIRE(torch::allclose(jac_row1, C.index({0})));
  REQUIRE(torch::allclose(jac_row2, C.index({1})));
}

TEST_CASE("jacrev", "[jacrev]")
{
  torch::Tensor x = torch::randn({3});
  x.set_requires_grad(true);
  torch::Tensor C = torch::randn({2, 3});

  torch::Tensor y = torch::matmul(C, x);
  torch::Tensor jac = jxt::jacrev(y, x);

  REQUIRE(torch::allclose(jac, C));
}

TEST_CASE("vmap jacrev", "[vmap][jacrev]")
{
  int64_t batch_size = 5;
  torch::Tensor x = torch::randn({batch_size, 3});
  torch::Tensor C = torch::randn({batch_size, 2, 3});

  torch::Tensor jac = jxt::vmap(
      [&](const torch::Tensor & C, const torch::Tensor & x)
      {
        torch::Tensor x_primal = x.clone().set_requires_grad(true);
        torch::Tensor y = torch::matmul(C, x);
        return jxt::jacrev(y, x);
      },
      {0, 0})(C, x);

  std::cout << jac << std::endl;
}
