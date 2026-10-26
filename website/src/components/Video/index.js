import React from 'react';
import {videoById} from '@site/src/content/videos';
import styles from './styles.module.css';

// An explainer video from src/content/videos.js, embedded from YouTube's
// privacy-enhanced domain (no cookies until the viewer plays the video).
export default function Video({id}) {
  const video = videoById(id);
  if (!video) {
    throw new Error(`Unknown video id "${id}"; add it to src/content/videos.js`);
  }
  return (
    <figure className={styles.frame}>
      <iframe
        src={`https://www.youtube-nocookie.com/embed/${video.youtube}?rel=0`}
        title={`${video.title} (video, ${video.duration})`}
        loading="lazy"
        allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
        referrerPolicy="strict-origin-when-cross-origin"
        allowFullScreen
      />
    </figure>
  );
}
