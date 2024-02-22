macro(create_symlinks TARGET_DIR HEADER_FILES)
    foreach(HEADER ${HEADER_FILES})
        get_filename_component(HEADER_NAME ${HEADER} NAME)
        set(SYMLINK_PATH ${TARGET_DIR}/${HEADER_NAME})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${HEADER} ${SYMLINK_PATH}
        )
    endforeach()
endmacro()