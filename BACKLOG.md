# Backlog

A running, user-managed list of ideas and future work for Planet Explorer,
grouped by priority. (The point-in-time engineering assessment lives in
`ARCHITECTURE_REVIEW.md`; the standards new work is held to live in
`ARCHITECTURE_GUIDELINES.md`.)

## Critical

_(none)_

## High

_(none)_

## Medium

_(none)_

## Low

- [ ] **Tour camera — keep motion generally forward.** During the guided tour, the
  camera's movement should always read as the viewer moving *forward*. A little
  randomized sideways slipping is fine, but it should never pull backward or drift
  directly sideways — generally forward, with some sideways slip.
  _Today the cruise phase picks a fully random great-circle drift direction
  (`tour.rs` → `begin_cruise` / `drift_axis`), so it can head backward or straight
  sideways relative to where the camera is looking._
