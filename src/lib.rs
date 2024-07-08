#![warn(clippy::restriction)]
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
to store cells that are out of bounds of the `[Grid<T>]` */

#[cfg(all(not(feature = "i32"), not(feature = "i64")))]
compile_error!("either feature \"i32\" or \"i64\" must be enabled");

#[cfg(all(feature = "i32", feature = "i64"))]
compile_error!("feature \"i32\" and feature \"i64\" cannot be enabled at the same time");

pub mod grid;
pub mod immutable;

#[cfg(feature = "bevy_reflect")]
use bevy_reflect::Reflect;
#[cfg(feature = "serde")]
use serde::{
    de::{self, Deserialize, Deserializer, MapAccess, Visitor},
    ser::{Serialize, SerializeStruct, Serializer},
};

use grid::Grid;
use immutable::Immutable;
use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::ops::{Index, IndexMut};

#[cfg(feature = "i32")]
type Scalar = i32;

#[cfg(feature = "i64")]
type Scalar = i64;


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
#[cfg_attr(feature = "bevy_reflect", derive(Reflect))]
pub struct Plane<T: Default, S: BuildHasher = RandomState> {
    grid: Grid<T>,
    // grid(x, y) = world(x, y) + offset
    offset: (Scalar, Scalar),
    map: HashMap<(Scalar, Scalar), T, S>,

    /// Is referenced by [`get`] function, because it requires immutable reference to `self`,
    /// and in case of the value being uninitialized, function should return reference to something with the lifetime of `self`.
    default_value: Immutable<T>,
}


#[cfg(feature = "serde")]
impl<'de, T: Default + Deserialize<'de>, H: Default + BuildHasher> Deserialize<'de> for Plane<T, H> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use std::marker::PhantomData;
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Offset,
            Size,
            Grid,
            Map,
        }

        struct PlaneVisitor<T, H> {
            _p: PhantomData<(T, H)>,
        }

        impl <'de, T: Default + Deserialize<'de>, H: Default + BuildHasher> Visitor<'de> for PlaneVisitor<T, H> {
            type Value = Plane<T, H>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("struct Grid")
            }
            
            fn visit_map<V>(self, mut accsess: V) -> Result<Plane<T, H>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut offset: Option<(Scalar, Scalar)> = None;
                let mut size: Option<(Scalar, Scalar)> = None;
                let mut grid = None;
                let mut map = None;
                while let Some(key) = accsess.next_key()? {
                    match key {
                        Field::Offset => {
                            if offset.is_some() {
                                return Err(de::Error::duplicate_field("offset"));
                            }
                            offset = Some(accsess.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(accsess.next_value()?);
                        }
                        Field::Grid => {
                            if grid.is_some() {
                                return Err(de::Error::duplicate_field("grid"));
                            }
                            grid = Some(accsess.next_value()?);
                        }
                        Field::Map => {
                            if map.is_some() {
                                return Err(de::Error::duplicate_field("map"));
                            }
                            map = Some(accsess.next_value()?);
                        }
                    }
                }

                let (x_min, y_min) = offset.ok_or_else(|| de::Error::missing_field("offset"))?;
                let map = map.ok_or_else(|| de::Error::missing_field("map"))?;
                
                if let Some(grid) = grid {
                    Ok(Plane::from_grid_and_hash_map(grid, map, -x_min, -y_min))
                } else if let Some((x_size, y_size)) = size {
                    Ok(Plane::from_hash_map(map, -x_min, -y_min, x_size - x_min, y_size - y_min))
                } else {
                    Err(de::Error::missing_field("grid or size"))
                }
            }
        }

        const FIELDS: &'static [&'static str] = &["cols", "data", "order"];
        deserializer.deserialize_struct("Grid", FIELDS, PlaneVisitor { _p: PhantomData })
    }
}

#[cfg(feature = "serde")]
impl<T: Default + Serialize, H: BuildHasher> Serialize for Plane<T, H> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("Plane", 3)?;
        state.serialize_field("offset", &self.offset)?;
        state.serialize_field("grid", &self.grid)?;
        state.serialize_field("map", &self.map)?;
        state.end()
    }
}


#[inline]
fn rows_cols(x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) -> (usize, usize) {
    ((x_max - x_min + 1) as usize, (y_max - y_min + 1) as usize)
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

impl<T: Default> Plane<T, RandomState> {
    #[inline]
    pub fn new(x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) -> Self {
        Self::default_hasher(x_min, y_min, x_max, y_max)
    }
}

impl<T: Default, S: Default + BuildHasher> Plane<T, S> {
    #[inline]
    pub fn default_hasher(x_min: Scalar, y_min: Scalar, x_max: Scalar, y_max: Scalar) -> Self {
        Self::with_hasher(x_min, y_min, x_max, y_max, Default::default())
    }
    #[inline]
    pub fn from_grid(grid: Grid<T>, x_min: Scalar, y_min: Scalar) -> Self {
        Self::from_grid_and_hash_map(grid, HashMap::default(), x_min, y_min)
    }
}


impl<T: Default, S: BuildHasher> Plane<T, S> {
    /// Returns [`Plane`] whose array-based grid is within specified bounds
    pub fn with_hasher(
        x_min: Scalar,
        y_min: Scalar,
        x_max: Scalar,
        y_max: Scalar,
        hasher: S,
    ) -> Self {
        let (rows, cols): (usize, usize) = rows_cols(x_min, y_min, x_max, y_max);
        Self::from_grid_and_hash_map(
            Grid::new(rows, cols),
            HashMap::with_hasher(hasher),
            x_min,
            y_min,
        )
    }

    #[inline]
    pub fn inner_grid(&self) -> &Grid<T> {
        &self.grid
    }
    #[inline]
    pub fn inner_grid_mut(&mut self) -> &mut Grid<T> {
        &mut self.grid
    }
    #[inline]
    pub fn inner_hash_map(&self) -> &HashMap<(Scalar, Scalar), T, S> {
        &self.map
    }
    #[inline]
    pub fn inner_hash_map_mut(&mut self) -> &mut HashMap<(Scalar, Scalar), T, S> {
        &mut self.map
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
    pub fn from_grid_with_hasher(grid: Grid<T>, x_min: Scalar, y_min: Scalar, hasher: S) -> Self {
        Self::from_grid_and_hash_map(grid, HashMap::with_hasher(hasher), x_min, y_min)
    }

    /// Creates instance of [`Plane<T>`] from [`Grid<T>`] and [`HashMap<(Scalar, Scalar), T>`]
    /// # Note
    /// Doesn't remove items from `map` if they are initialized and overlapping with `grid`. Their existence will be ignored.
    /// When you are calling [`inner_hash_map`], [`inner_hash_map_mut`], [`iter_all`], [`iter_all_mut`] or [`into_iter_all`]
    /// those values may or may not still exist in the hash map.
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
    /// Returns [`default`] value if uninitialized element is being accessed.
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
    /// Returns [`default`] value if uninitialized element is being accessed.
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
    /// Returns [`default`] value if uninitialized element is being accessed.
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
        let (rows, cols) = rows_cols(x_min, y_min, x_max, y_max);
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

    /// Iterates over all the items within the rectangle area inclusively.
    /// Returns [`default`] value if uninitialized element is being accessed.
    /// Order of iteration is deterministic, but can change in future versions.
    pub fn iter_rect(
        &self,
        x_min: Scalar,
        y_min: Scalar,
        x_max: Scalar,
        y_max: Scalar,
    ) -> impl Iterator<Item = ((Scalar, Scalar), &T)> {
        (x_min..=x_max)
            .map(move |x| (y_min..=y_max).map(move |y| ((x, y), self.get(x, y))))
            .flatten()
    }

    /// Mutably iterates over all the items within the rectangle area inclusively.
    /// Returns [`default`] value if uninitialized element is being accessed.
    /// Order of iteration is deterministic, but can change in future versions .
    pub fn iter_rect_mut(
        &mut self,
        x_min: Scalar,
        y_min: Scalar,
        x_max: Scalar,
        y_max: Scalar,
    ) -> impl Iterator<Item = ((Scalar, Scalar), &mut T)> {
        let plane: *mut _ = self;

        (x_min..=x_max)
            .map(move |x| {
                (y_min..=y_max).map(move |y| {
                    (
                        (x, y),
                        // SAFETY: `plane` is only set to `self` which has suitable lifetime
                        // none of the references returned twice since no pare of (x, y) is output twice
                        // and get_mut is guaranteed to return unique memory address for unique (x, y) pare
                        unsafe { &mut *plane }.get_mut(x, y),
                    )
                })
            })
            .flatten()
    }

    /// Iterate over all the elements stored inside the grid and hashmap. May return value from HashMap even if it is overlapping with Grid   
    pub fn iter_all(&self) -> impl Iterator<Item = ((Scalar, Scalar), &T)> {
        self.grid
            .indexed_iter()
            .map(move |((x, y), elem)| {
                (
                    (x as Scalar - self.offset.0, y as Scalar - self.offset.1),
                    elem,
                )
            })
            .chain(self.map.iter().map(|(vec, elem)| (*vec, elem)))
    }

    /// Mutably iterate over all the elements stored inside the grid and hashmap. May return value from HashMap even if it is overlapping with Grid   
    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = ((Scalar, Scalar), &mut T)> {
        let offset = self.offset;

        self.grid
            .indexed_iter_mut()
            .map(move |((x, y), elem)| ((x as Scalar - offset.0, y as Scalar - offset.1), elem))
            .chain(self.map.iter_mut().map(|(vec, elem)| (*vec, elem)))
    }

    /// Iterate over all the elements stored inside the grid and hashmap. May return value from HashMap even if it is overlapping with Grid   
    pub fn into_iter_all(self) -> impl Iterator<Item = ((Scalar, Scalar), T)> {
        self.grid
            .into_iter_indexed()
            .map(move |((x, y), elem)| {
                (
                    (x as Scalar - self.offset.0, y as Scalar - self.offset.1),
                    elem,
                )
            })
            .chain(self.map.into_iter())
    }
}


impl<T, S: BuildHasher> Plane<Option<T>, S> {
    /// Iterate over all the initialized elements
    pub fn iter(&self) -> impl Iterator<Item = ((Scalar, Scalar), &T)> {
        self.grid
            .indexed_iter()
            .filter_map(move |((x, y), elem)| {
                elem.as_ref().map(|el| {
                    (
                        (x as Scalar - self.offset.0, y as Scalar - self.offset.1),
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
                    .map(|el| ((x as Scalar - offset.0, y as Scalar - offset.1), el))
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
                        (x as Scalar - self.offset.0, y as Scalar - self.offset.1),
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
    #[inline]
    fn from(value: Grid<T>) -> Self {
        Self::from_grid(value, 0, 0)
    }
}


impl<T: Default, S: BuildHasher> From<HashMap<(Scalar, Scalar), T, S>> for Plane<T, S> {
    #[inline]
    fn from(value: HashMap<(Scalar, Scalar), T, S>) -> Self {
        Self::from_hash_map(value, 0, 0, 0, 0)
    }
}

impl <T: Default, S: BuildHasher> Index<(Scalar, Scalar)> for Plane<T, S> {
    type Output = T;
    #[inline]
    fn index(&self, (x, y): (Scalar, Scalar)) -> &Self::Output {
        self.get(x, y)
    }
}

impl <T: Default, S: BuildHasher> IndexMut<(Scalar, Scalar)> for Plane<T, S> {
    #[inline]
    fn index_mut(&mut self, (x, y): (Scalar, Scalar)) -> &mut Self::Output {
        self.get_mut(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::Plane;
    use super::grid::{grid, Grid};
    use std::collections::HashMap;

    #[test]
    fn test_iter() {
        let grid: Grid<Option<i32>> = grid![
            [Some(1), Some(2)]
            [Some(3), Some(4)]
        ];
        let mut hash_map = HashMap::new();
        hash_map.insert((23, 40), Some(19));
        hash_map.insert((40, 40), Some(13));
        let plane: Plane<Option<i32>> = Plane::from_grid_and_hash_map(grid, hash_map, 2, 2);

        let mut elements: Vec<_> = plane.iter().map(|(a, v)| (a, *v)).collect();
        elements.sort_by(|(_, v1), (_, v2)| v1.cmp(v2));

        assert_eq!(
            elements,
            vec![
                ((2, 2), 1),
                ((2, 3), 2),
                ((3, 2), 3),
                ((3, 3), 4),
                ((40, 40), 13),
                ((23, 40), 19),
            ]
        );
    }

    #[test]
    fn test_iter_mut() {
        let grid: Grid<Option<i32>> = grid![
            [Some(1), Some(2)]
            [Some(3), Some(4)]
        ];
        let mut hash_map = HashMap::new();
        hash_map.insert((23, 40), Some(19));
        hash_map.insert((40, 40), Some(13));

        let mut plane: Plane<Option<i32>> = Plane::from_grid_and_hash_map(grid, hash_map, 2, 2);

        for (_, elem) in plane.iter_mut() {
            *elem += 1;
        }

        let mut elements: Vec<_> = plane.iter().map(|(a, v)| (a, *v)).collect();
        elements.sort_by(|(_, v1), (_, v2)| v1.cmp(v2));

        assert_eq!(
            elements,
            vec![
                ((2, 2), 2),
                ((2, 3), 3),
                ((3, 2), 4),
                ((3, 3), 5),
                ((40, 40), 14),
                ((23, 40), 20),
            ]
        );
    }

    #[test]
    fn test_into_iter() {
        let grid: Grid<Option<i32>> = grid![
            [Some(1), Some(2)]
            [Some(3), Some(4)]
        ];
        let mut hash_map = HashMap::new();
        hash_map.insert((23, 40), Some(19));
        hash_map.insert((40, 40), Some(13));
        let plane: Plane<Option<i32>> = Plane::from_grid_and_hash_map(grid, hash_map, 2, 2);

        let mut elements: Vec<_> = plane.into_iter().collect();
        elements.sort_by(|(_, v1), (_, v2)| v1.cmp(v2));

        assert_eq!(
            elements,
            vec![
                ((2, 2), 1),
                ((2, 3), 2),
                ((3, 2), 3),
                ((3, 3), 4),
                ((40, 40), 13),
                ((23, 40), 19),
            ]
        );
    }

    #[test]
    fn test_default_initialization() {
        let plane: Plane<Option<()>> = Plane::default();
        let elements: Vec<_> = plane.into_iter().collect();

        assert_eq!(elements.len(), 0);
    }

    #[test]
    fn test_inserting_element() {
        let mut plane = Plane::new(1, 1, 4, 4);

        assert_eq!(*plane.get(1, 1), None);
        assert_eq!(*plane.get(4, 4), None);
        assert_eq!(*plane.get(1, 100), None);

        assert_eq!(plane.insert(Some(5), 1, 1), None);
        assert_eq!(*plane.get(1, 1), Some(5));

        assert_eq!(plane.insert(Some(6), -1, -1), None);
        assert_eq!(*plane.get(-1, -1), Some(6));

        assert_eq!(plane.insert(Some(8), -1, -1), Some(6));
        assert_eq!(*plane.get(-1, -1), Some(8));

        assert_eq!(plane.insert(Some(7), 4, 4), None);
        assert_eq!(*plane.get(4, 4), Some(7));

        assert_eq!(plane.insert(Some(9), 4, 4), Some(7));
        assert_eq!(*plane.get(4, 4), Some(9));
    }

    #[test]
    fn test_get_element_mut() {
        let mut plane = Plane::new(1, 1, 4, 4);
        plane.insert(Some(8), 1, 1);
        if let Some(elem) = plane.get_mut(1, 1) {
            *elem = 10;
        }
        assert_eq!(*plane.get(1, 1), Some(10));
    }

    #[test]
    fn test_iter_rect() {
        let mut plane = Plane::new(1, 1, 4, 4);
        plane.insert((1, 1), 1, 1);
        plane.insert((2, 1), 2, 1);
        plane.insert((3, 1), 3, 1);
        plane.insert((4, 1), 4, 1);
        plane.insert((5, 1), 5, 1);

        plane.insert((1, 2), 1, 2);
        plane.insert((2, 2), 2, 2);
        plane.insert((3, 2), 3, 2);
        plane.insert((4, 2), 4, 2);
        plane.insert((5, 2), 5, 2);

        plane.insert((1, 3), 1, 3);
        plane.insert((2, 3), 2, 3);
        plane.insert((3, 3), 3, 3);
        plane.insert((4, 3), 4, 3);
        plane.insert((5, 3), 5, 3);

        plane.insert((1, 4), 1, 4);
        plane.insert((2, 4), 2, 4);
        plane.insert((3, 4), 3, 4);
        plane.insert((4, 4), 4, 4);
        plane.insert((5, 4), 5, 4);

        plane.insert((1, 5), 1, 5);
        plane.insert((2, 5), 2, 5);
        plane.insert((3, 5), 3, 5);
        plane.insert((4, 5), 4, 5);
        plane.insert((5, 5), 5, 5);

        let mut value: Vec<_> = plane.iter_rect(2, 2, 6, 5).map(|(a, v)| (a, *v)).collect();
        value.sort_by(
            |((x1, y1), _), ((x2, y2), _)| {
                if x1 == x2 {
                    y1.cmp(y2)
                } else {
                    x1.cmp(x2)
                }
            },
        );

        assert_eq!(
            value,
            vec![
                ((2, 2), (2, 2)),
                ((2, 3), (2, 3)),
                ((2, 4), (2, 4)),
                ((2, 5), (2, 5)),
                ((3, 2), (3, 2)),
                ((3, 3), (3, 3)),
                ((3, 4), (3, 4)),
                ((3, 5), (3, 5)),
                ((4, 2), (4, 2)),
                ((4, 3), (4, 3)),
                ((4, 4), (4, 4)),
                ((4, 5), (4, 5)),
                ((5, 2), (5, 2)),
                ((5, 3), (5, 3)),
                ((5, 4), (5, 4)),
                ((5, 5), (5, 5)),
                ((6, 2), (0, 0)),
                ((6, 3), (0, 0)),
                ((6, 4), (0, 0)),
                ((6, 5), (0, 0)),
            ]
        );
    }

    #[test]
    fn test_iter_rect_mut() {
        let mut plane = Plane::new(1, 1, 4, 4);
        plane.insert((1, 1), 1, 1);
        plane.insert((2, 1), 2, 1);
        plane.insert((3, 1), 3, 1);
        plane.insert((4, 1), 4, 1);
        plane.insert((5, 1), 5, 1);

        plane.insert((1, 2), 1, 2);
        plane.insert((2, 2), 2, 2);
        plane.insert((3, 2), 3, 2);
        plane.insert((4, 2), 4, 2);
        plane.insert((5, 2), 5, 2);

        plane.insert((1, 3), 1, 3);
        plane.insert((2, 3), 2, 3);
        plane.insert((3, 3), 3, 3);
        plane.insert((4, 3), 4, 3);
        plane.insert((5, 3), 5, 3);

        plane.insert((1, 4), 1, 4);
        plane.insert((2, 4), 2, 4);
        plane.insert((3, 4), 3, 4);
        plane.insert((4, 4), 4, 4);
        plane.insert((5, 4), 5, 4);

        plane.insert((1, 5), 1, 5);
        plane.insert((2, 5), 2, 5);
        plane.insert((3, 5), 3, 5);
        plane.insert((4, 5), 4, 5);
        plane.insert((5, 5), 5, 5);

        plane.iter_rect_mut(3, 2, 6, 5).for_each(|(_, (xv, yv))| {
            *xv += 10;
            *yv += 20;
        });

        let mut value: Vec<_> = plane.iter_rect(2, 2, 7, 5).map(|(a, v)| (a, *v)).collect();
        value.sort_by(
            |((x1, y1), _), ((x2, y2), _)| {
                if x1 == x2 {
                    y1.cmp(y2)
                } else {
                    x1.cmp(x2)
                }
            },
        );

        assert_eq!(
            value,
            vec![
                ((2, 2), (2, 2)),
                ((2, 3), (2, 3)),
                ((2, 4), (2, 4)),
                ((2, 5), (2, 5)),
                ((3, 2), (13, 22)),
                ((3, 3), (13, 23)),
                ((3, 4), (13, 24)),
                ((3, 5), (13, 25)),
                ((4, 2), (14, 22)),
                ((4, 3), (14, 23)),
                ((4, 4), (14, 24)),
                ((4, 5), (14, 25)),
                ((5, 2), (15, 22)),
                ((5, 3), (15, 23)),
                ((5, 4), (15, 24)),
                ((5, 5), (15, 25)),
                ((6, 2), (10, 20)),
                ((6, 3), (10, 20)),
                ((6, 4), (10, 20)),
                ((6, 5), (10, 20)),
                ((7, 2), (0, 0)),
                ((7, 3), (0, 0)),
                ((7, 4), (0, 0)),
                ((7, 5), (0, 0)),
            ]
        );
    }
}
