cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerMRView)

set(EXTENSION_HOMEPAGE "https://www.mrtrix.org")
set(EXTENSION_CATEGORY "Diffusion")
set(EXTENSION_CONTRIBUTORS "Nicolas Tempier")
set(EXTENSION_DESCRIPTION "mrview-like visualisation of MRtrix3 tractograms (.tck) and FODs, with live ROI filtering.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(MRView)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
