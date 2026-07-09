if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans metrics infra_logger
        gtest_main gtest
    )
    gtest_discover_tests(ucmshared.test)

    if(UCM_RUNTIME_ASCEND_FAMILY)
        add_executable(ascend_register_limit_test
            example/trans/ascend_register_limit_test.cc
        )
        target_link_libraries(ascend_register_limit_test PRIVATE trans)
    endif()
endif()
