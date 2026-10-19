import Layout from '@theme-original/DocItem/Layout';
import Head from '@docusaurus/Head';
import {useLocation} from '@docusaurus/router';
import DraftBanner from '@site/src/components/Cookbook/DraftBanner';
import {cookbookPage} from '@site/src/content/cookbooks';

// Wraps every docs page. On a cookbook page the wrapper adds the working-draft
// banner to each chapter and, until the cookbook is listed as indexed in
// website/cookbook-visibility.json, a noindex tag.
export default function LayoutWrapper(props) {
  const {pathname} = useLocation();
  const page = cookbookPage(pathname);
  return (
    <>
      {page && page.draft && (
        <Head>
          <meta name="robots" content="noindex" />
        </Head>
      )}
      {page && page.draft && !page.isIndex && <DraftBanner entry={page.entry} />}
      <Layout {...props} />
    </>
  );
}
