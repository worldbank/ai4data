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
      description="Explainer videos of the AI for Data – Data for AI program, with links to the documentation and code of each method and a transcript of the narration.">
      <main className={styles.page}>
        <Heading as="h1">Videos</Heading>
        <p className={styles.lede}>
          This page lists the program&rsquo;s explainer videos. Each entry gives the length, the area of
          the <Link to="/hub">implementation hub</Link>, links to the documentation and code of the
          method, and a transcript of the narration.
        </p>
        <p className={styles.note}>
          The narration is read by a synthetic voice (Kokoro, voice af_heart). The videos are hosted
          on YouTube and embedded from its privacy-enhanced domain.
        </p>

        {videos.map((v) => (
          <section key={v.id} className={styles.video}>
            <Heading as="h2" id={v.id}>{v.title}</Heading>
            <p className={styles.meta}>
              {v.area} · {v.duration} · {v.published}
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
