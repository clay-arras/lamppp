find_package(Doxygen)
if (DOXYGEN_FOUND)
  set(DOXYGEN_INPUT_DIRS "${CMAKE_SOURCE_DIR}/csrc/src ${CMAKE_SOURCE_DIR}/csrc/include ${CMAKE_CURRENT_SOURCE_DIR}")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set(PROJECT_NAME ${CMAKE_PROJECT_NAME})

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
  )

  doxygen_add_docs(
    docs
    "${CMAKE_SOURCE_DIR}/csrc/src" "${CMAKE_SOURCE_DIR}/csrc/include" "${CMAKE_CURRENT_SOURCE_DIR}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  )
endif()

add_subdirectory(examples)
