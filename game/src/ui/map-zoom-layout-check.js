const DEFAULT_VIEWPORT = Object.freeze({ width: 3840, height: 2108 });
const DEFAULT_TOLERANCE = 1;

// This contract is executed against the live Online game page by browser QA.
export function readMapZoomLayoutGeometry(documentRef = document, windowRef = window) {
  const box = (selector) => {
    const element = documentRef.querySelector(selector);
    if (!element) throw new Error(`Missing layout element: ${selector}`);
    const rect = element.getBoundingClientRect();
    return {
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      clientWidth: element.clientWidth,
      clientHeight: element.clientHeight,
      scrollWidth: element.scrollWidth,
      scrollHeight: element.scrollHeight,
      transform: windowRef.getComputedStyle(element).transform,
    };
  };

  const viewport = box("#boardViewport");
  const map = box("#mapImage");
  return {
    innerWidth: windowRef.innerWidth,
    innerHeight: windowRef.innerHeight,
    documentScrollHeight: documentRef.documentElement.scrollHeight,
    bodyScrollHeight: documentRef.body.scrollHeight,
    game: box(".game-view"),
    topbar: box(".topbar"),
    viewport,
    map,
    sidePanel: box(".side-panel"),
    controls: box("#mapZoomControls"),
    images: {
      total: documentRef.images.length,
      complete: [...documentRef.images].filter((image) => image.complete && image.naturalWidth > 0).length,
    },
    mapAspectRatio: map.width / map.height,
  };
}

export function validateMapZoomLayout(
  geometry,
  { expectedViewport = DEFAULT_VIEWPORT, tolerance = DEFAULT_TOLERANCE } = {},
) {
  const failures = [];
  check(geometry.innerWidth === expectedViewport.width, `expected innerWidth ${expectedViewport.width}, got ${geometry.innerWidth}`);
  check(geometry.innerHeight === expectedViewport.height, `expected innerHeight ${expectedViewport.height}, got ${geometry.innerHeight}`);
  check(geometry.documentScrollHeight <= geometry.innerHeight + tolerance, `document scrollHeight ${geometry.documentScrollHeight} exceeds viewport ${geometry.innerHeight}`);
  check(geometry.bodyScrollHeight <= geometry.innerHeight + tolerance, `body scrollHeight ${geometry.bodyScrollHeight} exceeds viewport ${geometry.innerHeight}`);
  check(geometry.game.scrollHeight <= geometry.game.clientHeight + tolerance, `game scrollHeight ${geometry.game.scrollHeight} exceeds clientHeight ${geometry.game.clientHeight}`);
  check(geometry.viewport.bottom <= geometry.innerHeight + tolerance, `map viewport bottom ${geometry.viewport.bottom} is clipped by viewport ${geometry.innerHeight}`);
  check(geometry.sidePanel.bottom <= geometry.innerHeight + tolerance, `side panel bottom ${geometry.sidePanel.bottom} is clipped by viewport ${geometry.innerHeight}`);
  check(geometry.viewport.scrollHeight <= geometry.viewport.clientHeight + tolerance, `fitted map requires vertical scrolling: ${geometry.viewport.scrollHeight}/${geometry.viewport.clientHeight}`);
  const verticalUnusedSpace = geometry.viewport.clientHeight - geometry.map.height;
  check(
    Math.abs(verticalUnusedSpace) <= tolerance,
    verticalUnusedSpace >= 0
      ? `map leaves ${verticalUnusedSpace}px vertical unused space in the viewport`
      : `map exceeds the viewport height by ${Math.abs(verticalUnusedSpace)}px`,
  );
  check(geometry.map.top >= geometry.viewport.top - tolerance, "map is clipped at the top");
  check(geometry.map.bottom <= geometry.viewport.bottom + tolerance, "map is clipped at the bottom");
  check(geometry.map.left >= geometry.viewport.left - tolerance, "map is clipped at the left");
  check(geometry.map.right <= geometry.viewport.right + tolerance, "map is clipped at the right");
  check(Math.abs(geometry.mapAspectRatio - 2448 / 1696) < 0.001, `map aspect ratio changed to ${geometry.mapAspectRatio}`);
  check(geometry.topbar.transform === "none", `topbar unexpectedly transformed: ${geometry.topbar.transform}`);
  check(geometry.sidePanel.transform === "none", `side panel unexpectedly transformed: ${geometry.sidePanel.transform}`);
  const viewportClientRight = geometry.viewport.left + geometry.viewport.clientWidth;
  check(
    geometry.controls.right <= viewportClientRight + tolerance,
    `zoom controls overlap the map viewport scrollbar: ${geometry.controls.right}/${viewportClientRight}`,
  );
  check(geometry.controls.top >= geometry.viewport.top, "zoom controls overlap above the map viewport");
  check(geometry.images.total > 0 && geometry.images.complete === geometry.images.total, `only ${geometry.images.complete}/${geometry.images.total} images loaded`);
  return failures;

  function check(condition, message) {
    if (!condition) failures.push(message);
  }
}

export function assertMapZoomLayout(options = {}) {
  const geometry = readMapZoomLayoutGeometry();
  const failures = validateMapZoomLayout(geometry, options);
  if (failures.length) {
    throw new Error(`Map zoom layout regression failed:\n- ${failures.join("\n- ")}\nGEOMETRY=${JSON.stringify(geometry)}`);
  }
  return geometry;
}
