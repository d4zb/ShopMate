# Error analysis

Configuration: shipped default. Split: `dev` (150 sessions).
Hit@10 0.9933 | MRR 0.9665 | MTTC 2.653 | TechnicalScore 0.9536

## Parsing health

- Openers whose `coarse_category` failed to resolve: **0/150**
- Follow-up turns that yielded no constraint: **8**

## Missed entirely (1)

### `public_0083` (buying, easy)
- target `B0BPMCJ1RD`, pool trajectory: `260 -> 54 -> 20 -> 20 -> 20 -> 20 -> 20 -> 20 -> 20 -> 20`
- turns: 10, category `Tees & Blouses Blouses & Button-Down Shirts`

## Found but ranked below 1 (5 worst)

### `public_0087` (browsing, medium)
- rank 7 at turn 10, pool trajectory: `288 -> 115 -> 47 -> 47 -> 47 -> 47 -> 47 -> 47 -> 47 -> 47`
- constraints recovered: 4, exhausted: True

### `public_0020` (buying, easy)
- rank 4 at turn 2, pool trajectory: `33 -> 6`
- constraints recovered: 3, exhausted: False

### `public_0058` (buying, easy)
- rank 4 at turn 3, pool trajectory: `23 -> 7 -> 5`
- constraints recovered: 4, exhausted: False

### `public_0194` (buying, easy)
- rank 3 at turn 3, pool trajectory: `43 -> 8 -> 3`
- constraints recovered: 4, exhausted: False

### `public_0126` (browsing, medium)
- rank 2 at turn 3, pool trajectory: `681 -> 20 -> 5`
- constraints recovered: 4, exhausted: False

