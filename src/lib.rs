#![warn(clippy::all)]
/*! # Two Dimensional Plane
Continuous 2D data structure representing infinite 2d plane.
The purpose of this crate is to provide a universal data structure that is faster
than a naive`HashMap<(i32, i32), T>`
solution.

This crate will always provide a 2D data structure. If you need three or more dimensions take a look at the
other libraries. The `grid` crate is a container for all kinds of data that implement [`Default`] trait.
You can use [`Option<T>`] to store any kind of data.
No other dependencies except for the std lib are used.
Most of the functions `std::Vec<T>` offer are also implemented in `grid` and slightly modified for a 2D data object.

# Memory layout
Uses [grid](https://docs.rs/grid/0.14.0/grid/) crate to store a dense chunk of the grid and `HashMap<(i32, i32), T>`
to store cells that are out of bounds fo the `[Grid<T>]` */

#[cfg(all(not(feature = "i32"), not(feature = "i64")))]
compile_error!("either feature \"i32\" or \"i64\" must be enabled");

#[cfg(all(feature = "i32", feature = "i64"))]
compile_error!("feature \"i32\" and feature \"i64\" cannot be enabled at the same time");

pub mod immutable;

use grid::{Grid, Order};
use immutable::Immutable;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};

#[cfg(feature = "i32")]
type Scalar = i32;

#[cfg(feature = "i64")]
type Scalar = i64;

trait GridTrait<T> {
    fn into_iter_indexed(self) -> impl Iterator<Item = ((usize, usize), T)>;
}

impl<T> GridTrait<T> for Grid<T> {
    fn into_iter_indexed(self) -> impl Iterator<Item = ((usize, usize), T)> {
        let order = self.order();
        let cols = self.cols();
        let rows = self.rows();

        self.into_vec()
            .into_iter()
            .enumerate()
            .map(move |(idx, i)| {
                let position = match order {
                    Order::RowMajor => (idx / cols, idx % cols),
                    Order::ColumnMajor => (idx % rows, idx / rows),
                };
                (position, i)
            })
    }
}

/// Stores elements of a certain type in a 2D grid structure on the whole 2D plane-2d, even in negative direction.
///
/// Uses [`Grid<T>`] type in a [grid](https://docs.rs/grid/0.14.0/grid/) crate
/// and a
#[cfg_attr(feature = "i32", doc = "`HashMap<(i32, i32), T>`")]
#[cfg_attr(feature = "i64", doc = "`HashMap<(i64, i64), T>`")]
///  to store data on the heap.
///
/// Data in [`Grid<T>`] is stored inside one dimensional array using [`Vec<T>`].
/// This is cash efficient, so it is recommended to store there dense regions of data,
/// but it is memory inefficient - it keeps memory for `rows * cols` cells,
/// so if there are only two cells in use - one is placed on coordinate `(0,0)` and other is on `(100,100)`,
/// there is space reserved for at least `10000` elements.
///
/// Using [`HashMap`] solves that problem - it stores data outside the grid bounds.
///
/// Note that if the size of the Grid is zero, this data structure is identical to the
#[cfg_attr(feature = "i32", doc = "`HashMap<(i32, i32), T>`,")]
#[cfg_attr(feature = "i64", doc = "`HashMap<(i64, i64), T>`,")]
/// and it'll be more effective to just use
#[cfg_attr(feature = "i32", doc = "`HashMap<(i32, i32), T>`,")]
#[cfg_attr(feature = "i64", doc = "`HashMap<(i64, i64), T>`")]
/// since in this case you'll get rid of any unnecessary checks.
///
/// `T` should implement [`Default`] trait, because the plain is infinitely large,
/// and you can access any point of it at any time.
/// Whenever uninitialized cell is accessed, default value is returned.
/// For optionally initialized data use [`Option<T>`].
///
/// The size limit for the grid is `rows * cols < usize`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Plane<T: Default, S: Default + BuildHasher = RandomState> {
    grid: Grid<T>,
    offset: (Scalar, Scalar),
    map: HashMap<(Scalar, Scalar), T, S>,

    /// Is referenced by [`get`] function, because it requires immutable reference to `self`,
    /// and in case of the value being uninitialized, function should return reference to something with the lifetime of `self`.
    default_value: Immutable<T>,
}

impl<T: Default, S: Default + BuildHasher> Default for Plane<T, S> {
    /// Creates new `Plane<T>` with internal `Grid<T>` of size 0.
    /// Basically identical to
    #[cfg_attr(feature = "i32", doc = "`HashMap<(i32, i32), T>`,")]
    #[cfg_attr(feature = "i64", doc = "`HashMap<(i64, i64), T>`,")]
    /// better use it instead, because Plane will be doing unnecessary comparisons
    fn default() -> Self {
        Self {
            grid: Grid::new(0, 0),
            offset: (0, 0),
            map: HashMap::default(),
            default_value: Immutable::default(),
        }
    }
}

impl<T: Default, S: Default + BuildHasher> Plane<T, S> {
    #[inline]
    pub fn rows_cols(x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) -> (usize, usize) {
        ((x_max - x_min + 1) as usize, (y_max - y_min + 1) as usize)
    }

    /// Returns [`Plane`] whose array-based grid is within specified bounds
    pub fn new(x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) -> Self {
        let (rows, cols): (usize, usize) = Self::rows_cols(x_min, y_min, x_max, y_max);
        Self::from_grid_and_hash_map(Grid::new(rows, cols), HashMap::default(), x_min, y_min)
    }

    #[inline]
    pub fn inner_grid(&self) -> &Grid<T> {
        &self.grid
    }

    #[inline]
    pub fn inner_hash_map(&self) -> &HashMap<(Scalar, Scalar), T, S> {
        &self.map
    }

    pub fn from_hash_map(
        map: HashMap<(Scalar, Scalar), T, S>,
        x_min: Scalar,
        y_min: Scalar,
        x_max: Scalar,
        y_max: Scalar,
    ) -> Self {
        let mut plane = Self::from_grid_and_hash_map(Grid::new(0, 0), map, 0, 0);
        plane.relocate_grid(x_min, x_max, y_min, y_max);
        plane
    }

    #[inline]
    pub fn from_grid(grid: Grid<T>, x_min: Scalar, y_min: Scalar) -> Self {
        Self::from_grid_and_hash_map(grid, HashMap::default(), x_min, y_min)
    }

    pub fn from_grid_and_hash_map(
        grid: Grid<T>,
        map: HashMap<(Scalar, Scalar), T, S>,
        x_min: Scalar,
        y_min: Scalar,
    ) -> Self {
        Self {
            grid,
            offset: (-x_min, -y_min),
            map,
            default_value: Immutable::default(),
        }
    }

    pub fn into_hash_map(self) -> HashMap<(Scalar, Scalar), T, S> {
        let mut map = self.map;

        for ((x, y), val) in self.grid.into_iter_indexed() {
            let vec = (x as Scalar - self.offset.0, y as Scalar - self.offset.1);
            map.insert(vec, val);
        }

        map
    }

    pub fn global_coordinates_from_grid(&self, x: usize, y: usize) -> (Scalar, Scalar) {
        let x = x as Scalar - self.offset.0;
        let y = y as Scalar - self.offset.1;
        (x, y)
    }

    pub fn grid_coordinates_from_global(&self, x: Scalar, y: Scalar) -> Option<(usize, usize)> {
        let x = x + self.offset.0;
        let y = y + self.offset.1;

        if 0 <= x && x < self.grid.rows() as Scalar && 0 <= y && y < self.grid.cols() as Scalar {
            Some((x as usize, y as usize))
        } else {
            None
        }
    }

    /// Returns a reference to an element that should be contained in `Grid<T>` container without performing bound checks.
    /// Generally not recommended, use with caution!
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    pub unsafe fn get_unchecked(&self, x: Scalar, y: Scalar) -> &T {
        let x = (x + self.offset.0) as usize;
        let y = (y + self.offset.1) as usize;

        self.grid.get_unchecked(x, y)
    }

    /// Returns a mutable reference to an element that should be contained in `Grid<T>` container without performing bound checks.
    /// Generally not recommended, use with caution!
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    pub unsafe fn get_unchecked_mut(&mut self, x: Scalar, y: Scalar) -> &mut T {
        let x = (x + self.offset.0) as usize;
        let y = (y + self.offset.1) as usize;

        self.grid.get_unchecked_mut(x, y)
    }

    /// Access a certain element on the plane-2d.
    /// Returns [`default`] value if an uninitialized element is accessed.
    pub fn get(&self, x: Scalar, y: Scalar) -> &T {
        if let Some((x, y)) = self.grid_coordinates_from_global(x, y) {
            // Safety: `grid_coordinates_from_global` is guaranteed to return Some
            // whenever the coords are within the bounds of internal grid
            unsafe { self.grid.get_unchecked(x, y) }
        } else {
            let val = self.map.get(&(x, y));

            val.unwrap_or(&self.default_value)
        }
    }

    /// Mutable access to a certain element on the plane-2d.
    /// Returns [`default`] value if an uninitialized element is tried to be accessed.
    pub fn get_mut(&mut self, x: Scalar, y: Scalar) -> &mut T {
        if let Some((x, y)) = self.grid_coordinates_from_global(x, y) {
            // Safety: `grid_coordinates_from_global` is guaranteed to return Some
            // whenever the coords are within the bounds of internal grid
            unsafe { self.grid.get_unchecked_mut(x, y) }
        } else {
            self.map.entry((x, y)).or_default()
        }
    }

    /// Insert element at the coordinate.
    /// Returns value that was stored in a plane-2d or [`default`] value if element was uninitialized.
    pub fn insert(&mut self, value: T, x: Scalar, y: Scalar) -> T {
        if let Some((x, y)) = self.grid_coordinates_from_global(x, y) {
            // Safety: safe since `within_grid_bounds` is guaranteed to return true
            // whenever the coords are within the bounds of internal grid
            std::mem::replace(unsafe { self.grid.get_unchecked_mut(x, y) }, value)
        } else {
            self.map.insert((x, y), value).unwrap_or_default()
        }
    }

    /// Changes position of  the base grid structure.
    /// Iterates over all elements in previous grid and all elements in new grid
    pub fn relocate_grid(&mut self, x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) {
        let (rows, cols) = Self::rows_cols(x_min, y_min, x_max, y_max);
        let mut new_grid = Grid::new(rows, cols);

        let old_grid = std::mem::replace(&mut self.grid, Grid::new(0, 0));

        for ((x, y), val) in old_grid.into_iter_indexed() {
            let vec = self.global_coordinates_from_grid(x, y);
            self.map.insert(vec, val);
        }

        for ((x, y), val) in new_grid.indexed_iter_mut() {
            let vec = self.global_coordinates_from_grid(x, y);
            *val = self.map.remove(&vec).unwrap_or_default();
        }

        self.grid = new_grid;
        self.offset = (-x_min, -y_min);
    }
}

impl<T, S: Default + BuildHasher> Plane<Option<T>, S> {
    /// Iterate over all the initialized elements
    pub fn iter(&self) -> impl Iterator<Item = ((Scalar, Scalar), &T)> {
        self.grid
            .indexed_iter()
            .filter_map(move |((x, y), elem)| {
                elem.as_ref().map(|el| {
                    (
                        (x as Scalar + self.offset.0, y as Scalar + self.offset.1),
                        el,
                    )
                })
            })
            .chain(
                self.map
                    .iter()
                    .filter_map(|(vec, elem)| elem.as_ref().map(|el| (*vec, el))),
            )
    }

    /// Mutably iterate over all the initialized elements
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ((Scalar, Scalar), &mut T)> {
        let offset = self.offset;

        self.grid
            .indexed_iter_mut()
            .filter_map(move |((x, y), elem)| {
                elem.as_mut()
                    .map(|el| ((x as Scalar + offset.0, y as Scalar + offset.1), el))
            })
            .chain(
                self.map
                    .iter_mut()
                    .filter_map(|(vec, elem)| elem.as_mut().map(|el| (*vec, el))),
            )
    }

    /// Consume plane-2d to get all the initialized elements
    pub fn into_iter(self) -> impl Iterator<Item = ((Scalar, Scalar), T)> {
        self.grid
            .into_iter_indexed()
            .filter_map(move |((x, y), elem)| {
                elem.map(|el| {
                    (
                        (x as Scalar + self.offset.0, y as Scalar + self.offset.1),
                        el,
                    )
                })
            })
            .chain(
                self.map
                    .into_iter()
                    .filter_map(|(vec, elem)| elem.map(|el| (vec, el))),
            )
    }
}

impl<T: Default, S: Default + BuildHasher> From<Grid<T>> for Plane<T, S> {
    fn from(value: Grid<T>) -> Self {
        Self::from_grid(value, 0, 0)
    }
}

impl<T: Default, S: Default + BuildHasher> From<HashMap<(Scalar, Scalar), T, S>> for Plane<T, S> {
    fn from(value: HashMap<(Scalar, Scalar), T, S>) -> Self {
        Self::from_hash_map(value, 0, 0, 0, 0)
    }
}
