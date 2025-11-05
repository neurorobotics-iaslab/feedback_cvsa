cmake_minimum_required(VERSION 3.0)
project(feedback_cvsa)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(catkin REQUIRED COMPONENTS 
             roscpp 
             std_msgs
             std_srvs
             dynamic_reconfigure
             rosneuro_msgs
             message_generation
             )

find_library(NEURODRAW_LIBRARIES neurodraw REQUIRED)
find_path(NEURODRAW_INCLUDE_DIRS neurodraw/Shape.h REQUIRED) 

find_library(NEUROCHRONO_LIBRARIES neurochrono REQUIRED)
find_path(NEUROCHRONO_INCLUDE_DIRS neurochrono/Timer.h REQUIRED)

find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
find_library(SNDFILE_LIBRARY NAMES sndfile)

find_path(AO_INCLUDE_DIR NAMES ao/ao.h)
find_library(AO_LIBRARY NAMES ao)

generate_dynamic_reconfigure_options(
  cfg/CVSA.cfg
)

add_service_files(
  FILES
  Repeat_trial.srv
)
add_message_files(
  FILES
    Trials_to_keep.msg
    AudioMessage.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS 
    include
  LIBRARIES 
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    std_msgs
    rosneuro_msgs
    message_runtime
    std_srvs
  DEPENDS
)


###########
## Build ##
###########

include_directories(include 
                    ${catkin_INCLUDE_DIRS}
                    ${NEURODRAW_INCLUDE_DIRS}
                    ${NEUROCHRONO_INCLUDE_DIRS}
                    ${AO_INCLUDE_DIRS}
                    ${SNDFILE_INCLUDE_DIRS}
                    )

add_library(${PROJECT_NAME}    
            src/CVSA_layout.cpp
            src/CVSA_layout_visual.cpp
            src/TrainingCVSA.cpp
            src/ControlCVSA.cpp
            src/TrialSequence.cpp
            src/TrainingCVSA_visual.cpp
            src/Autopilot.cpp
            )

target_link_libraries(${PROJECT_NAME} 
                      ${catkin_LIBRARIES}
                      ${NEURODRAW_LIBRARIES}
                      ${NEUROCHRONO_LIBRARIES}
                      ${SNDFILE_LIBRARY}
                      ${AO_LIBRARY}
                      )

add_dependencies(${PROJECT_NAME} 
                 ${catkin_EXPORTED_TARGETS}
                 ${PROJECT_NAME}_gencfg
)

#################
## Executables ##
#################
add_executable(trainingCVSA_node src/trainingCVSA_node.cpp)
target_link_libraries(trainingCVSA_node  ${PROJECT_NAME}
                                    ${catkin_LIBRARIES}
                                    ${NEURODRAW_LIBRARIES}
                                    ${NEUROCHRONO_LIBRARIES}
                                    ${SNDFILE_LIBRARY}
                                    ${AO_LIBRARY}
                                    )
add_executable(controlCVSA_node src/controlCVSA_node.cpp)
target_link_libraries(controlCVSA_node ${PROJECT_NAME}
                                   ${catkin_LIBRARIES}
                                   ${NEURODRAW_LIBRARIES}
                                   ${NEUROCHRONO_LIBRARIES}
                                   )

add_executable(trainingCVSA_visual_node src/trainingCVSA_visual_node.cpp)
target_link_libraries(trainingCVSA_visual_node ${PROJECT_NAME}
                                   ${catkin_LIBRARIES}
                                   ${NEURODRAW_LIBRARIES}
                                   ${NEUROCHRONO_LIBRARIES}
                                   )
