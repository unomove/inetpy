"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
load("//engine/build:isaac.bzl", "isaac_py_app")

isaac_py_app(
    name = "inet",
    srcs = [
        "__init__.py",
        "inet.py",
    ],
    data = [
        "//apps:py_init",
        "//apps/spot/inetpy/models",
        "//apps/spot/subgraphs:d435i_subgraph",
        "inet.app.json",
        "//messages:differential_base_proto",
        "//messages:state_proto",
        "//messages:math_proto",
        "//messages:image_proto",
        "//packages/record_replay/apps:record_subgraph",
        "//packages/navigation/apps:differential_base_commander_subgraph",
        "//packages/navigation/apps:differential_base_imu_odometry_subgraph"
    ],
    modules=[
        "//apps/spot:spot_interface_module",
    ],
    deps = ["//engine/pyalice"],
)