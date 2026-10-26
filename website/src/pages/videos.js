import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Video from '@site/src/components/Video';
import {videos} from '@site/src/content/videos';
import styles from './videos.module.css';

// The catalogue of explainer videos. Records are in src/content/videos.js;
// the same component embeds a video on the documentation page of its method.

function VideoLink({link}) {
  return link.href ? <a href={link.href}>{link.label}</a> : <Link to={link.to}>{link.label}</Link>;
}

export default function Videos() {
  return (
    <Layout
      title="Videos"
      description="Short videos that explain the methods and tools of the AI for Data – Data for AI program on real examples, with links to the documentation and code and a transcript of each video.">
      <main className={styles.page}>
        <Heading as="h1">Videos</Heading>
        <p className={styles.lede}>
          The program makes short videos that explain its methods and tools. Each video takes one
          method, shows it at work on a real example, and explains the problem that the method
          addresses. A video is an introduction to the method. The documentation describes the method
          in full, and the code that implements it is linked from each entry.
        </p>
        <p className={styles.lede}>
          A transcript of the narration follows each video, for readers who prefer text or who want
          to quote a passage.
        </p>

        {videos.map((v) => (
          <section key={v.id} className={styles.video}>
            <Heading as="h2" id={v.id}>{v.title}</Heading>
            <p className={styles.meta}>
              {v.area} · {v.duration} · {v.published}
              {v.voice === 'synthetic' && ' · Narrated by a synthetic voice'}
            </p>
            <Video id={v.id} />
            <p className={styles.summary}>{v.summary}</p>
            <ul className={styles.links}>
              {v.links.map((l) => (
                <li key={l.label}>
                  <VideoLink link={l} />
                </li>
              ))}
              <li>
                <a href={`https://youtu.be/${v.youtube}`}>Open on YouTube</a>
              </li>
            </ul>
            <details className={styles.transcript}>
              <summary>Transcript</summary>
              {v.transcript.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </details>
          </section>
        ))}
      </main>
    </Layout>
  );
}
