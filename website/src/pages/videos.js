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
          The AI for Data – Data for AI program publishes short explainer videos alongside its
          documentation. Each video introduces one method or tool from the program, such as
          Proof-Carrying Numbers, through a worked example on public World Bank data. It shows the
          problem that the method solves and then walks through how the method works, step by step.
        </p>
        <p className={styles.lede}>
          Each entry below links to the documentation, which describes the method in full, and to
          the code that implements it. The full text of the narration is under each video.
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
