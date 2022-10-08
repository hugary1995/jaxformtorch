#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <cmath>
#include "jaxformtorch.h"

TEST_CASE("wrap a function", "[function]")
{
  std::function<double(double)> f = (double (*)(double)) & std::sin;
  jxt::vmap<double, double> f_mapped(f);

  REQUIRE(f_mapped(5.5) == Approx(std::sin(5.5)));
}
