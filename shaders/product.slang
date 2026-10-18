// C = A * B for row-major float32 matrices: A is rows x depth, B is depth x columns.
//
// Each 256-thread workgroup computes a 64 x 64 tile of C; each thread owns a 4 x 4 block of adjacent
// rows and columns in registers, so every value it reads from shared memory feeds four
// multiply-adds. Slabs of A and B, sixteen deep, pass through shared memory. When the shapes allow
// 16-byte loads, every thread fetches one float4 of A and one of B per slab, and fetches the next
// slab while the current one is being multiplied. Measured on an Apple M1 through MoltenVK at about
// 400-680 GFLOP/s for n >= 1024, against 52 for 8 x 8 blocks (register spills).
//
// Avoid `thread`, `device`, `constant` and other Metal keywords as names: MoltenVK translates to
// Metal source, and a surviving name breaks the shader compile.

struct Arguments {
    float* left;
    float* right;
    float* output;
    uint rows;
    uint columns;
    uint depth;
    uint first;
};

[[vk::push_constant]] Arguments arguments;

static const uint tile = 64;
static const uint slab = 16;
static const uint block = 4;
static const uint side = tile / block;
static const uint threads = side * side;

groupshared float lefts[slab][tile];
groupshared float rights[slab][tile];

// Rows [first, first + 64 * groups.y) of C; the host splits tall products into several dispatches.
[shader("compute")]
[numthreads(16, 16, 1)]
void product(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    const uint rows = arguments.rows;
    const uint columns = arguments.columns;
    const uint depth = arguments.depth;
    const uint top = arguments.first + group.y * tile;
    const uint left = group.x * tile;
    const uint member = local.y * side + local.x;

    float sums[block][block];
    for (uint down = 0; down < block; ++down) {
        for (uint over = 0; over < block; ++over) {
            sums[down][over] = 0.0;
        }
    }

    if (depth % 4 == 0 && columns % 4 == 0) {
        // One float4 of A (along depth) and one of B (along columns) per thread per slab.
        const uint across = member / (slab / 4);
        const uint along = (member % (slab / 4)) * 4;
        const uint down = member / (tile / 4);
        const uint over = (member % (tile / 4)) * 4;
        float4* lefthand = (float4*)arguments.left;
        float4* righthand = (float4*)arguments.right;

        // The slab after the current one: upcoming[0] from A, upcoming[1] from B.
        float4 upcoming[2];
        upcoming[0] = top + across < rows && along < depth ? lefthand[((top + across) * depth + along) / 4] : float4(0.0);
        upcoming[1] = down < depth && left + over < columns ? righthand[(down * columns + left + over) / 4] : float4(0.0);

        for (uint start = 0; start < depth; start += slab) {
            lefts[along][across] = upcoming[0].x;
            lefts[along + 1][across] = upcoming[0].y;
            lefts[along + 2][across] = upcoming[0].z;
            lefts[along + 3][across] = upcoming[0].w;
            rights[down][over] = upcoming[1].x;
            rights[down][over + 1] = upcoming[1].y;
            rights[down][over + 2] = upcoming[1].z;
            rights[down][over + 3] = upcoming[1].w;
            GroupMemoryBarrierWithGroupSync();

            // Fetch the following slab while this one is multiplied.
            const uint ahead = start + slab;
            upcoming[0] = top + across < rows && ahead + along < depth ? lefthand[((top + across) * depth + ahead + along) / 4] : float4(0.0);
            upcoming[1] = ahead + down < depth && left + over < columns ? righthand[((ahead + down) * columns + left + over) / 4] : float4(0.0);

            multiply(local, sums);
            GroupMemoryBarrierWithGroupSync();
        }
    } else {
        for (uint start = 0; start < depth; start += slab) {
            for (uint step = 0; step < (tile * slab) / threads; ++step) {
                const uint item = member + step * threads;

                const uint row = item / slab;
                const uint inner = item % slab;
                lefts[inner][row] = top + row < rows && start + inner < depth ? arguments.left[(top + row) * depth + start + inner] : 0.0;

                const uint lower = item / tile;
                const uint column = item % tile;
                rights[lower][column] = start + lower < depth && left + column < columns ? arguments.right[(start + lower) * columns + left + column] : 0.0;
            }
            GroupMemoryBarrierWithGroupSync();
            multiply(local, sums);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    for (uint down = 0; down < block; ++down) {
        const uint row = top + local.y * block + down;
        if (row >= rows) continue;
        for (uint over = 0; over < block; ++over) {
            const uint column = left + local.x * block + over;
            if (column < columns) {
                arguments.output[row * columns + column] = sums[down][over];
            }
        }
    }
}

void multiply(uint3 local, inout float sums[block][block]) {
    for (uint inner = 0; inner < slab; ++inner) {
        // This thread's four rows of A and four columns of B at depth `inner`.
        float column[block];
        float row[block];
        for (uint offset = 0; offset < block; ++offset) {
            column[offset] = lefts[inner][local.y * block + offset];
            row[offset] = rights[inner][local.x * block + offset];
        }
        for (uint down = 0; down < block; ++down) {
            for (uint over = 0; over < block; ++over) {
                sums[down][over] = fma(column[down], row[over], sums[down][over]);
            }
        }
    }
}
