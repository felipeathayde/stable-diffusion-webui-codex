# apps/interface/public Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Static assets served directly by Vite (favicons, robots, manifest). No bundling or transformation occurs here.

## Notes
- Place only static files that need to be served as-is. For processed assets, use the `src/` pipeline.
