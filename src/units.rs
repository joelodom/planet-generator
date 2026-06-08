//! Real-world unit formatting for the HUD/log. The render world uses abstract
//! "units" (see [`crate::planet::METERS_PER_UNIT`]); this converts them to metres
//! and formats as metric (default) or US customary.

use crate::planet::METERS_PER_UNIT;

const FEET_PER_METER: f64 = 3.280_84;
const FEET_PER_MILE: f64 = 5280.0;
const METERS_PER_KM: f64 = 1000.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Units {
    Metric,
    Us,
}

impl Units {
    pub fn parse(s: &str) -> Option<Units> {
        match s.to_ascii_lowercase().as_str() {
            "metric" | "si" | "m" => Some(Units::Metric),
            "us" | "imperial" | "customary" => Some(Units::Us),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Units::Metric => "metric",
            Units::Us => "US",
        }
    }
}

/// Convert render units to metres.
pub fn to_meters(units: f32) -> f64 {
    units as f64 * METERS_PER_UNIT as f64
}

/// Format a distance/altitude (given in render units) for display.
pub fn distance(units_value: f32, sys: Units) -> String {
    let m = to_meters(units_value);
    match sys {
        Units::Metric => {
            if m.abs() >= METERS_PER_KM {
                format!("{:.1} km", m / METERS_PER_KM)
            } else {
                format!("{:.0} m", m)
            }
        }
        Units::Us => {
            let ft = m * FEET_PER_METER;
            if ft.abs() >= FEET_PER_MILE {
                format!("{:.1} mi", ft / FEET_PER_MILE)
            } else {
                format!("{:.0} ft", ft)
            }
        }
    }
}

/// Format an elevation (render units, may be negative) as plain m/ft.
pub fn elevation(units_value: f32, sys: Units) -> String {
    let m = to_meters(units_value);
    match sys {
        Units::Metric => format!("{:.0} m", m),
        Units::Us => format!("{:.0} ft", m * FEET_PER_METER),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_metric_and_us() {
        // 100 render units = 1000 m = 1 km.
        assert_eq!(distance(100.0, Units::Metric), "1.0 km");
        // 1000 render units = 10 km ≈ 6.2 mi.
        assert!(distance(1000.0, Units::Us).ends_with("mi"));
        // small distances stay in m / ft.
        assert!(distance(5.0, Units::Metric).ends_with('m'));
        assert!(distance(5.0, Units::Us).ends_with("ft"));
    }

    #[test]
    fn parse_flags() {
        assert_eq!(Units::parse("US"), Some(Units::Us));
        assert_eq!(Units::parse("metric"), Some(Units::Metric));
        assert_eq!(Units::parse("nonsense"), None);
    }
}
