<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

# Documentation Contribution Guide

This guide explains how to contribute to the ThermOHL project documentation.

## Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

## Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/thermohl.git
   cd thermohl
   ```
3. Create a virtual environment:
   ```bash
   uv venv venv
   source venv/bin/activate  # Linux/MacOS
   # or
   .\venv\Scripts\activate  # Windows
   ```
4. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

## Documentation Structure

The documentation is organized in the `thermohl-docs/docs` directory as follows:
```
thermohl-docs/
└── docs/
    ├── index.md          # Homepage
    ├── user-guide/       # User guide
    ├── docstring/        # Docstring documentation
    └── assets/          # Documentation assets
        ├── images/      # Image files
        └── diagrams/    # Diagram files
```

## Managing Assets

### Images and Diagrams

1. Place all images in `thermohl-docs/docs/assets/images/`
   - Use descriptive filenames (e.g., `temperature-profile.png`)
   - Prefer SVG format for diagrams and PNG/JPG for photos
   - Keep file sizes reasonable (optimize if needed)

2. Reference images in your Markdown:
   ```markdown
   ![Alt text](assets/images/filename.png)
   ```

3. For diagrams:
   - Place source files in `thermohl-docs/docs/assets/diagrams/`
   - Export to both source format and image format
   - Include a README.md in the diagrams folder explaining tools used

### Best Practices for Assets

- Keep asset filenames lowercase with underscore
- Use relative paths in Markdown links
- Optimize images for web (use tools like ImageOptim)
- Document any special requirements for diagrams
- Keep the assets directory organized by type
- Update asset references when moving files

## Writing Documentation

1. Create a new branch for your contribution:
   ```bash
   git checkout -b doc/description-of-your-contribution
   ```

2. Modify or create Markdown files in the `thermohl-docs/docs/` directory

3. Preview your documentation locally:
   ```bash
   mkdocs serve
   ```
   Visit http://127.0.0.1:8000 to see the result

4. Commit your changes:
   ```bash
   git add .
   git commit -m "docs: description of your changes"
   ```

5. Push your changes:
   ```bash
   git push origin doc/description-of-your-contribution
   ```

6. Create a Pull Request on GitHub

## Best Practices

- Use clear and concise language
- Add code examples when relevant
- Check spelling and grammar
- Follow the existing documentation style
- Use hierarchical headings (##, ###, etc.)
- Add links to other parts of the documentation when relevant

## Formatting

Use Markdown syntax to write documentation. For a complete reference, visit the [official Markdown Guide](https://www.markdownguide.org/).

## Verification

Before submitting your PR, make sure that:

1. The documentation builds without errors:
   ```bash
   mkdocs build
   ```

2. The style is consistent:
   ```bash
   black thermohl-docs/docs/
   ```

3. There are no typographical errors:
   ```bash
   # Install codespell if needed
   pip install codespell
   codespell thermohl-docs/docs/
   ```

## Questions and Help

If you have questions or need help:
- Open an issue on GitHub
- Join our discussion channel
- Contact the project maintainers

Thank you for contributing to ThermOHL's documentation! 