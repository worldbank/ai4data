import styles from './styles.module.css';

const CLIENTS = ['Claude', 'ChatGPT', 'Custom Agent'];

export default function McpDiagram() {
  return (
    <div className={styles.diagram}>
      <div className={styles.block}>
        <span className={styles.blockLabel}>Before — one integration per client</span>
        {CLIENTS.map((c) => (
          <div className={styles.chain} key={c}>
            <span className={styles.box}>{c}</span>
            <span className={styles.arrow}>→</span>
            <span className={styles.boxMuted}>custom plugin</span>
            <span className={styles.arrow}>→</span>
            <span className={styles.boxMuted}>WDI API</span>
          </div>
        ))}
      </div>

      <div className={styles.block}>
        <span className={styles.blockLabel}>With MCP — one server, every client</span>
        <div className={styles.chain}>
          <div className={styles.stack}>
            {CLIENTS.map((c) => (
              <span className={styles.box} key={c}>
                {c}
              </span>
            ))}
          </div>
          <span className={styles.arrow}>→</span>
          <span className={styles.boxAccent}>MCP Client</span>
          <span className={styles.arrow}>→</span>
          <span className={styles.boxAccent}>MCP Server</span>
          <span className={styles.arrow}>→</span>
          <span className={styles.box}>WDI API / Catalog</span>
        </div>
      </div>
    </div>
  );
}
