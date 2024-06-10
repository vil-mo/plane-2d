use std::ops::Deref;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Helper struct that insures that inner value can't be mutated even if user has mutable access to it.
#[derive(Debug, Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Immutable<T>(T);

impl <T> Immutable<T> {
	#[inline]
	pub fn new(value: T) -> Self {
		Self(value)
	}

	#[inline]
	pub fn get(&self) -> &T {
		&self.0
	}
}

impl <T> Deref for Immutable<T> {
	type Target = T;
	
	#[inline]
	fn deref(&self) -> &Self::Target {
		self.get()
	}
}
