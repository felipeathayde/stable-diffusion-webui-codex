# apps/interface/src/styles Overview
<!-- tags: frontend, styles, tailwind -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-03
Status: Active

## Purpose
- Component/view-specific CSS modules layered on top of the global Tailwind tokens defined in `styles.css`.

## Notes
- Follow the semantic class naming guidance in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Add new files per component/view rather than embedding large rule sets in shared sheets.
- 2025-12-03: Refiner styling gains an embedded/dense variant for the hires nested card; Highres card styles include a nested refiner section separator.
