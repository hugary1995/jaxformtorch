#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <cmath>
#include "jaxformtorch.h"

TEST_CASE("wrap a function", "[function]")
{
  // Urgh, this cast is unfortunately necessary to help the compiler disambiguate std::sin as it is
  // an overloaded function...
  std::function<double(double)> sin_double = (double (*)(double)) & std::sin;

  REQUIRE(jxt::vmap<double, double>(sin_double)(5.5) == Approx(std::sin(5.5)));
}
