set(TBB_INSTALL OFF CACHE INTERNAL "" FORCE)
set(TBB_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(TBB_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(TBB_FMT_EXTERNAL ON CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_TBB_NAME tbb)
    set(DEP_TBB_TAG v2022.3.0)
    set(DEP_TBB_GIT_URLS
        https://github.com/uxlfoundation/oneTBB.git
        https://gitcode.com/gh_mirrors/on/oneTBB.git
    )
    include(helper.cmake)
    find_reachable_git_url(REACHABLE_URL DEP_TBB_GIT_URLS)
    include(FetchContent)
    message(STATUS "Fetching ${DEP_TBB_NAME}(${DEP_TBB_TAG}) from ${REACHABLE_URL}")
    FetchContent_Declare(${DEP_TBB_NAME} GIT_REPOSITORY ${REACHABLE_URL} GIT_TAG ${DEP_TBB_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEP_TBB_NAME})
else()
    add_subdirectory(tbb)
endif()
