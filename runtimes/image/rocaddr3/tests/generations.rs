// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native surface layout contracts for `AddrLib2` and `AddrLib3` families.

#![allow(clippy::unwrap_used, clippy::panic)]

use rocaddr3::{ResourceType, SurfaceMapper, SurfaceRequest, compute, compute_linear};

fn request(family: u32, resource_type: ResourceType) -> SurfaceRequest {
    SurfaceRequest {
        family,
        chip_revision: 1,
        gb_addr_config: 0,
        bits_per_pixel: 32,
        width: 64,
        height: 32,
        slices: 4,
        mip_levels: 5,
        resource_type,
    }
}

#[test]
fn each_supported_address_family_produces_complete_mip_storage() {
    for family in [
        0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x94, 0x95, 0x96, 0x97, 0x98, 0x9a,
    ] {
        for resource in [ResourceType::Texture2D, ResourceType::Texture3D] {
            let request = request(family, resource);
            let layout = compute(request)
                .unwrap_or_else(|error| panic!("family {family:x}, {resource:?}: {error:?}"));
            assert_eq!(layout.mips.len(), 5);
            assert!(layout.alignment.is_power_of_two());
            assert!(layout.size >= 64 * 32 * 4 * 4);
            assert!(layout.pitch >= request.width);
            assert!(layout.swizzle_mode != 0);
            for mip in &layout.mips {
                assert!(mip.offset < layout.size);
            }
        }
    }
}

#[test]
fn family_linear_layout_matches_its_row_and_slice_pitches() {
    for family in [0x8d, 0x8f, 0x91, 0x98] {
        let request = SurfaceRequest {
            mip_levels: 1,
            ..request(family, ResourceType::Texture2D)
        };
        let layout = compute_linear(request, 0).unwrap();
        assert_eq!(layout.swizzle_mode, 0);
        assert_eq!(layout.pitch, 64);
        assert_eq!(layout.slice_size, 64 * 32 * 4);
        assert_eq!(layout.size, 64 * 32 * 4 * 4);
        let single = SurfaceRequest {
            slices: 1,
            ..request
        };
        let explicit = compute_linear(single, 128).unwrap();
        assert_eq!(explicit.pitch, 128);
        assert!(explicit.size >= 128 * 32 * 4);
    }
}

#[test]
fn linear_mapper_preserves_row_and_slice_pitches() {
    for (width, slices) in [(17, 1), (17, 2), (64, 1), (64, 2)] {
        for requested_pitch in [0, 64, 96, 128] {
            let request = SurfaceRequest {
                width,
                slices,
                mip_levels: 1,
                ..request(0x98, ResourceType::Texture2D)
            };
            let layout = compute_linear(request, requested_pitch).unwrap();
            let mut mapper = SurfaceMapper::new(request, &layout).unwrap();
            let pitch = if requested_pitch == 0 {
                layout.mips[0].pitch
            } else {
                requested_pitch
            };
            for slice in 0..slices {
                for row in [0, 1, request.height - 1] {
                    let mut offsets = [0; 2];
                    mapper.map_row(0, 3, row, slice, &mut offsets).unwrap();
                    let expected = u64::from(slice) * layout.slice_size
                        + (u64::from(row) * u64::from(pitch) + 3) * 4;
                    assert_eq!(
                        offsets,
                        [expected, expected + 4],
                        "width {width}, pitch {requested_pitch}, row {row}, slice {slice}"
                    );
                }
            }
            for (x, y, slice) in [(width - 1, 0, 0), (0, request.height, 0), (0, 0, slices)] {
                assert_eq!(
                    mapper.map_row(0, x, y, slice, &mut [0; 2]),
                    Err(rocaddr3::Error::NoLayout)
                );
            }
        }
    }
}
