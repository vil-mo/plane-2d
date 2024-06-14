 # Two Dimensional Plane
Continuous 2D data structure, infinitely big.
The purpose of this crate is to provide a universal data structure that is faster
than a naive`HashMap<(i32, i32), T>`
solution.

This crate will always provide a 2D data structure. If you need three or more dimensions take a look at the
other libraries. The `grid` crate is a container for all kinds of data that implement `Default` trait.
You can use `Option<T>` to store any kind of data.
No other dependencies except for the std lib are used.
Most of the functions `std::Vec<T>` offer are also implemented in `grid` and slightly modified for a 2D data object.

# Memory layout
Uses [grid](https://docs.rs/grid/0.14.0/grid/) crate to store a dense chunk of the grid and `HashMap<(i32, i32), T>`
to store cells that are out of bounds of the `Grid<T>`
