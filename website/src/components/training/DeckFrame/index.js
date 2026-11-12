import React, {useEffect, useRef, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {useColorMode} from '@docusaurus/theme-common';
import styles from './styles.module.css';

// The HTML slide deck of a training session, shown in a frame on the Slides page. The
// deck lays out a 16:9 slide with a navigation bar under it, and in a window narrower
// than 820 pixels it becomes one scrolling page. The frame therefore keeps a fixed
// window of WIDTH by HEIGHT pixels and is scaled to the width of the page. On a page
// narrower than NARROW pixels the frame is not scaled, and the deck scrolls.
const WIDTH = 1280;
const HEIGHT = 760;
const NARROW = 560;

export default function DeckFrame({src, title}) {
  const url = useBaseUrl(src);
  const {colorMode} = useColorMode();
  const box = useRef(null);
  const frame = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const measure = () => setWidth(box.current ? box.current.clientWidth : 0);
    measure();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  // The deck follows the colour mode of the site.
  const applyTheme = () => {
    try {
      frame.current.contentDocument.documentElement.dataset.theme = colorMode;
    } catch (e) {}
  };
  useEffect(applyTheme, [colorMode, width > 0]);

  const fullScreen = () => {
    const el = frame.current;
    if (el && el.requestFullscreen) el.requestFullscreen().catch(() => {});
  };

  const narrow = width > 0 && width < NARROW;
  const scale = width / WIDTH;
  return (
    <>
      <p className="training-actions">
        <a className="button button--secondary button--sm" href={url} target="_blank" rel="noopener">
          Open in a new tab
        </a>
        <button type="button" className="button button--secondary button--sm" onClick={fullScreen}>
          Full screen
        </button>
      </p>
      <div
        ref={box}
        className={narrow ? styles.scrolling : styles.scaled}
        style={narrow || !width ? undefined : {height: Math.round(HEIGHT * scale)}}>
        {width > 0 && (
          <iframe
            ref={frame}
            src={url}
            title={title}
            onLoad={applyTheme}
            allow="fullscreen"
            allowFullScreen
            style={narrow ? undefined : {width: WIDTH, height: HEIGHT, transform: `scale(${scale})`}}
          />
        )}
      </div>
    </>
  );
}
