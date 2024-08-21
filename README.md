# Two Dimensional Plane
Models continuous, infinitely big (within integer and storage limits) 2D data structure.
The purpose of this crate is to provide a data structure that is faster
than a `HashMap<(i32, i32), T>` in specific scenarios and provides better API 
for working with 2D plane.

This crate will always provide a 2D data structure. 
The `Plane<T>` type is a container for all kinds of data that implement `Default` trait.
You can use `Option<T>` to store optionally initialized data.

No other dependencies except for the std lib are used, 
besides dependencies hidden behind feature flags.

# Memory layout
Uses almost exact copy of [grid](https://docs.rs/grid/0.14.0/grid/) crate to use `Grid<T>` type. 
Stores a dense chunk of the plane in `Vec<T>` (`Grid<T>`, provided by copy of the `grid` crate) 
and `HashMap<(i32, i32), T>` to store cells that are out of bounds of the `Grid<T>`.
Unlike `HashMap<(i32, i32), T>`, two allocations are being done. 
