# * Macro to create symbolic links
#
# This macro creates a symbolic link in target_dir for every file in files.
macro(CREATE_SYMLINKS target_dir files)
  foreach(file ${files})
    get_filename_component(file_name ${file} NAME)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${file}
                            ${target_dir}/${file_name})
  endforeach()
endmacro()
