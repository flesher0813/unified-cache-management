set(CCQ_INSTALL OFF CACHE INTERNAL "" FORCE)
set(CCQ_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(CCQ_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_CCQ_NAME concurrentqueue)
    set(DEP_CCQ_TAG v1.0.4)
    set(DEP_CCQ_GIT_URLS
        https://github.com/cameron314/concurrentqueue.git
        https://gitcode.com/GitHub_Trending/co/concurrentqueue.git
    )
    include(helper.cmake)
    find_reachable_git_url(REACHABLE_URL DEP_CCQ_GIT_URLS)
    include(FetchContent)
    message(STATUS "Fetching ${DEP_CCQ_NAME}(${DEP_CCQ_TAG}) from ${REACHABLE_URL}")
    FetchContent_Declare(${DEP_CCQ_NAME} GIT_REPOSITORY ${REACHABLE_URL} GIT_TAG ${DEP_CCQ_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEP_CCQ_NAME})
else()
    add_subdirectory(concurrentqueue)
endif()
