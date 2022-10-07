add_executable(unit_tests
      test.cxx
)

# Libraries and headers
target_link_libraries(unit_tests Catch2::Catch2)
target_include_directories(unit_tests PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(unit_tests ${TORCH_LIBRARIES})

# Register with CTest
include(CTest)
include(Catch)

catch_discover_tests(unit_tests)
