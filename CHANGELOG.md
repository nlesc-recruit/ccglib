# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- Support for FP8 e4m3 data type ([#24](https://github.com/nlesc-recruit/ccglib/pull/24))
- Hardware support checks for FP8 ([#35](https://github.com/nlesc-recruit/ccglib/pull/35))
- Implement scaling with alpha/beta ([#31](https://github.com/nlesc-recruit/ccglib/pull/31))
- Add support for bfloat16 ([#20](https://github.com/nlesc-recruit/ccglib/pull/20))
- Add start-to-end pipeline class ([#17](https://github.com/nlesc-recruit/ccglib/pull/17))
- Add compatibility with HIP 7 ([#26](https://github.com/nlesc-recruit/ccglib/pull/26))
- Set PATH and CXX for HIP test workflow ([#28](https://github.com/nlesc-recruit/ccglib/pull/28))
- Guard tf32 for architectures that don't support it ([#23](https://github.com/nlesc-recruit/ccglib/pull/23))
- Add CI for tests on GPU hardware ([#1](https://github.com/nlesc-recruit/ccglib/pull/1))

### Changed

- Move direction and complex axis location arguments from packing.Run to constructor ([#25](https://github.com/nlesc-recruit/ccglib/pull/25))
- Release preparation ([#34](https://github.com/nlesc-recruit/ccglib/pull/34))
- Implement syncwarp for HIP<7 ([#29](https://github.com/nlesc-recruit/ccglib/pull/29))
- Make type_selector compatible with sm70 (Tesla V100) ([#21](https://github.com/nlesc-recruit/ccglib/pull/21))
- Cleanup type selector ([#11](https://github.com/nlesc-recruit/ccglib/pull/11))
- Fix compilation of benchmark in HIP mode when host compiler != hipcc ([#18](https://github.com/nlesc-recruit/ccglib/pull/18))
- Fix benchmark ([#12](https://github.com/nlesc-recruit/ccglib/pull/12))

## [0.1.0] - 2025-09-30

First release.
