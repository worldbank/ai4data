import {useCallback, useEffect, useRef, useState} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {useColorMode} from '@docusaurus/theme-common';
import styles from './ai-readiness-assessment.module.css';

// The framework summary is a self-contained page in static/assessment-framework.
// It is shown inside the site layout so the site header stays on screen.
function FrameworkFrame() {
  const src = useBaseUrl('/assessment-framework/');
  const {colorMode} = useColorMode();
  const frame = useRef(null);
  const [hash, setHash] = useState('');
  const [headerHidden, setHeaderHidden] = useState(false);
  const detach = useRef(null);

  useEffect(() => {
    setHash(window.location.hash);
  }, []);

  // Keep the embedded page on the same light or dark theme as the site.
  const syncTheme = useCallback(() => {
    const doc = frame.current && frame.current.contentDocument;
    if (doc && doc.documentElement) {
      doc.documentElement.setAttribute('data-theme', colorMode);
    }
  }, [colorMode]);

  useEffect(syncTheme, [syncTheme]);

  // The page scrolls inside the frame. Hide the site header on the way down,
  // and show it again on the way up, at the top of the page, or when the
  // pointer reaches the top edge.
  useEffect(() => {
    document.body.classList.add('ai-readiness-page');
    return () => {
      document.body.classList.remove('ai-readiness-page');
      document.body.classList.remove('ai-readiness-header-hidden');
    };
  }, []);

  useEffect(() => {
    document.body.classList.toggle('ai-readiness-header-hidden', headerHidden);
  }, [headerHidden]);

  const watchScroll = useCallback(() => {
    if (detach.current) {
      detach.current();
      detach.current = null;
    }
    const win = frame.current && frame.current.contentWindow;
    if (!win) {
      return;
    }
    let last = win.scrollY;
    const onScroll = () => {
      const y = win.scrollY;
      const delta = y - last;
      if (y < 40 || delta < -6) {
        setHeaderHidden(false);
      } else if (delta > 6) {
        setHeaderHidden(true);
      }
      last = y;
    };
    const onMove = (e) => {
      if (e.clientY < 8) {
        setHeaderHidden(false);
      }
    };
    win.addEventListener('scroll', onScroll, {passive: true});
    win.document.addEventListener('mousemove', onMove, {passive: true});
    detach.current = () => {
      win.removeEventListener('scroll', onScroll);
      win.document.removeEventListener('mousemove', onMove);
    };
  }, []);

  useEffect(() => () => detach.current && detach.current(), []);

  const onLoad = useCallback(() => {
    syncTheme();
    watchScroll();
  }, [syncTheme, watchScroll]);

  // The frame can finish loading before React attaches onLoad (server-rendered
  // markup), so check once after mounting.
  useEffect(() => {
    const doc = frame.current && frame.current.contentDocument;
    if (doc && doc.readyState === 'complete' && doc.body) {
      onLoad();
    }
  }, [onLoad]);

  return (
    <iframe
      ref={frame}
      className={clsx(styles.frame, headerHidden && styles.frameFull)}
      title="World Bank Group AI-readiness assessment framework"
      src={`${src}${hash}`}
      onLoad={onLoad}
    />
  );
}

export default function AiReadinessAssessment() {
  return (
    <Layout
      title="AI-readiness assessment framework"
      description="The World Bank Group AI-readiness assessment framework for national statistical organizations assesses the institution and the data products and services it provides.">
      <FrameworkFrame />
    </Layout>
  );
}
