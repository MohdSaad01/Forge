# Forge Development Workflow

## Control loop
1. Read current documentation and inspect code.
2. Select the current milestone.
3. Implement a vertical slice.
4. Add tests with the implementation.
5. Run tests and manual verification.
6. Compare against acceptance criteria.
7. Update affected documentation.
8. Report results.
9. Human reviews and commits coherent work manually.
10. Proceed to the next milestone.

## AI-assisted development
Claude Code may write production code and tests. The project owner remains responsible for understanding behavior, reviewing results, running the software, and making product decisions.

## Scope control
Do not add unrelated features because they appear useful. Record meaningful architectural changes in an ADR when they affect established boundaries.

## Git
No automated commits or pushes. Use coherent commits rather than streak-driven commit spam.
