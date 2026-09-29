import {useState} from 'react';
import styles from './styles.module.css';

const QUERY = 'income inequality data for Kenya';

export default function QueryToggle() {
  const [tab, setTab] = useState('search');
  const [copied, setCopied] = useState(false);

  const mcpPayload = `→ tools/call  search_indicators
  { "query": "${QUERY}" }

← { "indicator": "SI.POV.GINI",
    "country": "KEN",
    "value": 38.5,
    "year": 2022 }`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(mcpPayload);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable — no-op.
    }
  };

  return (
    <div className={styles.widget}>
      <div className={styles.tabs} role="tablist" aria-label="Query method">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'search'}
          className={tab === 'search' ? styles.tabActive : styles.tab}
          onClick={() => setTab('search')}>
          Natural Language Search
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'mcp'}
          className={tab === 'mcp' ? styles.tabActive : styles.tab}
          onClick={() => setTab('mcp')}>
          MCP Tool Call
        </button>
      </div>

      {tab === 'search' ? (
        <div className={styles.panel}>
          <div className={styles.searchBar}>
            <span className={styles.searchIcon} aria-hidden="true">
              &#8981;
            </span>
            <span className={styles.searchQuery}>{QUERY}</span>
          </div>
          <div className={styles.resultCard}>
            <div className={styles.resultTop}>
              <span className={styles.resultIndicator}>SI.POV.GINI</span>
              <span className={styles.resultBadge}>Gini index</span>
            </div>
            <div className={styles.resultBottom}>
              <span>Kenya</span>
              <span className={styles.resultDot} aria-hidden="true">
                &middot;
              </span>
              <span>2022</span>
              <span className={styles.resultDot} aria-hidden="true">
                &middot;
              </span>
              <span className={styles.resultValue}>38.5</span>
            </div>
          </div>
          <p className={styles.caption}>
            No exact keyword match required — the search understands intent.
          </p>
        </div>
      ) : (
        <div className={styles.panel}>
          <pre className={styles.code}>{mcpPayload}</pre>
          <div className={styles.panelFoot}>
            <p className={styles.caption}>
              Same question, same answer — issued as a tool call any MCP
              client can make.
            </p>
            <button type="button" className={styles.copyButton} onClick={handleCopy}>
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
