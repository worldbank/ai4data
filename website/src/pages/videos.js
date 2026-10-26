import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Video from '@site/src/components/Video';
import {videos} from '@site/src/content/videos';
import styles from './videos.module.css';

// The catalogue of the program's videos, grouped by kind. Records are in
// src/content/videos.js; the same component embeds a video on other pages.

function VideoLink({link}) {
  return link.href ? <a href={link.href}>{link.label}</a> : <Link to={link.to}>{link.label}</Link>;
}

// The kinds of video, in the order the page lists them. A kind with no
// videos is left out.
const KINDS = [
  ['concepts', 'Concepts'],
  ['methods', 'Program methods and tools'],
  ['walkthroughs', 'Walkthroughs'],
];

function VideoEntry({v}) {
  return (
    <article className={styles.video}>
      <Heading as="h3" id={v.id}>{v.title}</Heading>
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
    </article>
  );
}

export default function Videos() {
  return (
    <Layout
      title="Videos"
      description="Videos of the AI for Data – Data for AI program: explainers of concepts behind AI applications that answer from data, explainers of the program’s methods and tools, and walkthroughs of its products and site.">
      <main className={styles.page}>
        <Heading as="h1">Videos</Heading>
        <p className={styles.lede}>
          This page collects the videos of the AI for Data – Data for AI program. They are of three
          kinds: explainers of general concepts behind AI applications that answer from data, such as
          retrieval-augmented generation, tools and agents; explainers of the methods and tools in
          the program&rsquo;s <Link to="/tools">tools catalogue</Link>, such as the AI-readiness
          assessment, the Metadata Editor, the Data360 MCP server and Proof-Carrying Numbers; and
          walkthroughs that show a product or a page of this site in use.
        </p>
        <p className={styles.lede}>
          The explainers work through real examples, mostly on public World Bank data and
          publications, and show the problem that a technique addresses before the steps that address
          it. Each entry links to the documentation and code that the video relates to, and the full
          text of the narration is under each video.
        </p>

        {KINDS.map(([kind, heading]) => {
          const list = videos.filter((v) => v.kind === kind);
          if (!list.length) return null;
          return (
            <section key={kind} className={styles.kind}>
              <Heading as="h2" id={kind}>{heading}</Heading>
              {list.map((v) => (
                <VideoEntry key={v.id} v={v} />
              ))}
            </section>
          );
        })}
      </main>
    </Layout>
  );
}
