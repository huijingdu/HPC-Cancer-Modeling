#ifndef DOMAIN_H
#define DOMAIN_H

// refits the bin grid around [lo, hi] plus a margin. returns the new bin count.
int fit_bin_grid(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z);

// derive this rank's band from the global cut array
void apply_x_cuts(const int * cut);

// Work balanced X partition
void assign_x_cuts_balanced(const float * x, const float * y, const float * z,
                            const int * ele_per_cell, int cell_no);

#endif // DOMAIN_H
