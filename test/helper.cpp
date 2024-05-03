#include <catch2/catch_test_macros.hpp>

#include "helper/helper.h"

namespace ccglib::test {

TEST_CASE("CeilDiv exact") { REQUIRE(ccglib::helper::ceildiv(10, 5) == 2); }

TEST_CASE("CeilDiv inexact") { REQUIRE(ccglib::helper::ceildiv(16, 5) == 4); }

} // namespace ccglib::test
