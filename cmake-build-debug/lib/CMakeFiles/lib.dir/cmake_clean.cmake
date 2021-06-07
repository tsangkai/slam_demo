file(REMOVE_RECURSE
  "liblib.a"
  "liblib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
