# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add rotation alignment axis for 3D registration method.

### Changed

- **BREAKING**: Use Z-axis as default angle alignment axis for
  **Keller3DRegistration** registration method.

## [0.2.0] - 2025-03-14

### Added

- Add official support for Python 3.10.
- Add parameter for increased vectorization within registration methods that
  rely on the Pseudo-Polar Fourier Transform.

### Changed

- Switch to the external Pseudo-Polar Fourier Transform package
  [ppft-py](https://github.com/jnk22/ppft-py).
- Use real-valued functions in the Pseudo-Polar Fourier Transform for improved
  performance and reduced memory usage.
- Update usage of deprecated `pytransform3d` functions.

### Fixed

- Improve error handling for translation estimation failures during
  shift/rotation flip resolution.

## [0.1.0] - 2025-01-30

### Added

- Initial release of the project.

[Unreleased]: https://github.com/jnk22/ndimreg/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jnk22/ndimreg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jnk22/ndimreg/releases/tag/v0.1.0
