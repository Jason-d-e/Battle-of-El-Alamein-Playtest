export const MAP_ZOOM_LEVELS = Object.freeze([0.75, 1, 1.25, 1.5, 1.75, 2]);
export const DEFAULT_MAP_ZOOM = 1;

/**
 * Clamps and snaps a persisted map zoom value to a supported UI step.
 *
 * @param {unknown} value Candidate zoom value.
 * @returns {number} Supported zoom multiplier.
 */
export function normalizeMapZoom(value) {
  if (value === null || value === undefined || value === "") return DEFAULT_MAP_ZOOM;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_MAP_ZOOM;
  return MAP_ZOOM_LEVELS.reduce((nearest, level) => (
    Math.abs(level - numeric) < Math.abs(nearest - numeric) ? level : nearest
  ), MAP_ZOOM_LEVELS[0]);
}

/**
 * Moves one step through the fixed map zoom scale.
 *
 * @param {unknown} current Current zoom value.
 * @param {number} direction Negative to zoom out, positive to zoom in.
 * @returns {number} Next supported zoom multiplier.
 */
export function stepMapZoom(current, direction) {
  const normalized = normalizeMapZoom(current);
  const currentIndex = MAP_ZOOM_LEVELS.indexOf(normalized);
  const offset = Number(direction) < 0 ? -1 : Number(direction) > 0 ? 1 : 0;
  const nextIndex = Math.max(0, Math.min(MAP_ZOOM_LEVELS.length - 1, currentIndex + offset));
  return MAP_ZOOM_LEVELS[nextIndex];
}

export function formatMapZoom(value) {
  return `${Math.round(normalizeMapZoom(value) * 100)}%`;
}

export function mapZoomStageSize({ width, height, zoom } = {}) {
  const normalized = normalizeMapZoom(zoom);
  return {
    width: Math.max(0, Math.round(finiteNumber(width) * normalized)),
    height: Math.max(0, Math.round(finiteNumber(height) * normalized)),
  };
}

/**
 * Calculates scroll offsets that keep the same map coordinate at the center
 * after the map surface changes scale.
 */
export function preserveMapViewportCenter({
  scrollLeft = 0,
  scrollTop = 0,
  clientWidth = 0,
  clientHeight = 0,
  currentZoom = DEFAULT_MAP_ZOOM,
  nextZoom = DEFAULT_MAP_ZOOM,
} = {}) {
  const current = normalizeMapZoom(currentZoom);
  const next = normalizeMapZoom(nextZoom);
  const width = finiteNumber(clientWidth);
  const height = finiteNumber(clientHeight);
  const centerX = (finiteNumber(scrollLeft) + width / 2) / current;
  const centerY = (finiteNumber(scrollTop) + height / 2) / current;
  return {
    scrollLeft: Math.max(0, centerX * next - width / 2),
    scrollTop: Math.max(0, centerY * next - height / 2),
  };
}

function finiteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}
