cmake_minimum_required(VERSION 2.8)

project(minimap2_gactx)

include(FindZLIB)

find_package(ZLIB REQUIRED)
find_path(ZLIB_INCLUDE_DIR zlib.h)
include_directories(${ZLIB_INCLUDE_DIR})

find_package (Threads)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_DEBUG_POSTFIX "d")

set(CMAKE_CXX_COMPILER "/opt/Xilinx/SDx/2017.4.op/bin/xcpp")

set (XILINX_LINK_LIBS libxilinxopencl.so)
link_directories(${XILINX_SDX}/runtime/lib/x86_64 ${LD_LIBRARY_PATH})

############################################### set import and export folder ###

set(IMPORT_FOLDER "${CMAKE_SOURCE_DIR}/import/manylinux")
message(STATUS "Import folder is ${IMPORT_FOLDER}")

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
message(STATUS "Export folder is ${CMAKE_INSTALL_PREFIX}")

################################################# define compilation targets ###

include_directories(${IMPORT_FOLDER}/include ${ZLIB_INCLUDE_DIRS})
link_directories(${IMPORT_FOLDER}/lib)

add_library(libminimap2
    align.c
    bseq.c
	chain.c
    esterr.c
    format.c
    hit.c
    index.c
    kalloc.c
    ksw2_extd2_sse.c
    ksw2_exts2_sse.c
    ksw2_extz2_sse.c
	ksw2_dispatch.c
    ksw2_ll_sse.c
    kthread.c
    map.c
    misc.c
    options.c
    pe.c
    sdust.c
    sketch.c
    splitidx.c
	gactx.cpp
)

add_executable(minimap2_gactx
    main.cpp
)

####################################################### compilation settings ###

target_link_libraries(libminimap2
    PRIVATE
        ${ZLIB_LIBRARIES}
)

target_link_libraries(minimap2_gactx
    PRIVATE
        libminimap2
)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2 -Wc++-compat -s -DHAVE_KALLOC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2 -Wc++-compat -s -DHAVE_KALLOC -DSDX_PLATFORM=xilinx_aws-vu9p-f1-04261818_dynamic_5_0 -D__USE_XOPEN2K8 -I${XILINX_SDX}/runtime/include/1_2/ -I${XILINX_VIVADO}/include/ -fmessage-length=0 -std=c++11")

if(WIN32)
    add_definitions(-DHAVE_STRUCT_TIMESPEC)
    add_definitions(-DPTW32_STATIC_LIB)
if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Intel")
    # Explicitly define __SSE2__ for msvc, as the compiler doesn't define it by default.
    add_definitions(-D__SSE2__)
endif()
    target_link_libraries(libminimap2
        PRIVATE
            debug     pthreadVC2d
            optimized pthreadVC2
    )
else()
    if(ARM_BUILD)
        add_definitions(
            -D__SSE2__
            -DKSW_SSE2_ONLY
            -D_FILE_OFFSET_BITS=64
            -fsigned-char
        )
        endif()
        # The 'm' is not a typo -- we're passing in the linker flag "-lm"
    target_link_libraries(libminimap2
        PRIVATE
            m
            ${CMAKE_THREAD_LIBS_INIT}
    )
endif()

############################################################# install target ###

file(GLOB IMPORT_HEADER_FILES "${IMPORT_FOLDER}/include/*.h")
file(GLOB IMPORT_LIB_FILES "${IMPORT_FOLDER}/lib/*.dll")

install(
    FILES minimap.h bseq.h ${IMPORT_HEADER_FILES}
    DESTINATION include
)
install(
    FILES ${IMPORT_LIB_FILES}
    DESTINATION lib
)
install(
    TARGETS minimap2_gactx
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    target_link_libraries (minimap2_gactx PRIVATE rt stdc++ ${XILINX_LINK_LIBS} ${ZLIB_LIBRARIES})
endif()

if(TARGET zlib::zlib)
    target_link_libraries(minimap2_gactx PRIVATE rt stdc++ ${XILINX_LINK_LIBS} zlib::zlib)
endif()

execute_process (
    COMMAND bash -c "${XILINX_SDX}/bin/emconfigutil --od . --nd 1  --platform ${AWS_PLATFORM}")