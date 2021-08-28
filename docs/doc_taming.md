# All files
## link: https://github.com/CompVis/taming-transformers/tree/master
<details>
<summary>
<a name='taming/modules/misc/coord.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py'>taming/modules/misc/coord.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/modules/misc/coord.py:CoordStage' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py#L3'>CoordStage</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/misc/coord.py:CoordStage:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py#L4'>CoordStage:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n_embed,<br>down_factor,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/misc/coord.py:CoordStage:eval' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py#L8'>CoordStage:eval</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/misc/coord.py:CoordStage:encode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py#L11'>CoordStage:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>c,<br></ul>
        <li>Docs:<br>        """fake vqmodel interface"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/misc/coord.py:CoordStage:decode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/misc/coord.py#L27'>CoordStage:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>c,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='scripts/extract_segmentation.py' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py'>scripts/extract_segmentation.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='scripts/extract_segmentation.py:rescale_bgr' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L20'>rescale_bgr</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_segmentation.py:run_model' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L67'>run_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>img,<br>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_segmentation.py:get_input' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L75'>get_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>batch,<br>k,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_segmentation.py:save_segmentation' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L83'>save_segmentation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>segmentation,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_segmentation.py:iterate_dataset' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L94'>iterate_dataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dataloader,<br>destpath,<br>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L26'>COCOStuffSegmenter</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L27'>COCOStuffSegmenter:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:forward' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L42'>COCOStuffSegmenter:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>upsample = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:_pre_process' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L49'>COCOStuffSegmenter:_pre_process</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:mean' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L54'>COCOStuffSegmenter:mean</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:std' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L59'>COCOStuffSegmenter:std</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='scripts/extract_segmentation.py:COCOStuffSegmenter:input_size' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_segmentation.py#L63'>COCOStuffSegmenter:input_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/losses/lpips.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py'>taming/modules/losses/lpips.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/losses/lpips.py:normalize_tensor' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L116'>normalize_tensor</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>eps = 1e-10,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/losses/lpips.py:spatial_average' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L121'>spatial_average</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>keepdim = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/lpips.py:LPIPS' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L11'>LPIPS</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/lpips.py:ScalingLayer' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L57'>ScalingLayer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/lpips.py:NetLinLayer' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L67'>NetLinLayer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/lpips.py:vgg16' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L76'>vgg16</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:LPIPS:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L13'>LPIPS:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>use_dropout = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:LPIPS:load_from_pretrained' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L27'>LPIPS:load_from_pretrained</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>name = "vgg_lpips",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:LPIPS:from_pretrained' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L33'>LPIPS:from_pretrained</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cls,<br>name = "vgg_lpips",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:LPIPS:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L41'>LPIPS:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br>target,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:ScalingLayer:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L58'>ScalingLayer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:ScalingLayer:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L63'>ScalingLayer:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>inp,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:NetLinLayer:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L69'>NetLinLayer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>chn_in,<br>chn_out = 1,<br>use_dropout = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:vgg16:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L77'>vgg16:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>requires_grad = False,<br>pretrained = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/lpips.py:vgg16:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/lpips.py#L100'>vgg16:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/coco.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py'>taming/data/coco.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/coco.py:Examples' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L12'>Examples</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/coco.py:CocoBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L22'>CocoBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/coco.py:CocoImagesAndCaptionsTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L151'>CocoImagesAndCaptionsTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/coco.py:CocoImagesAndCaptionsValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L164'>CocoImagesAndCaptionsValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:Examples:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L13'>Examples:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size = 256,<br>random_crop = False,<br>interpolation = "bicubic",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L24'>CocoBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size = None,<br>dataroot = "",<br>datajson = "",<br>onehot_segmentation = False,<br>use_stuffthing = False,<br>crop_size = None,<br>force_no_crop = False,<br>given_files = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L93'>CocoBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoBase:preprocess_image' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L96'>CocoBase:preprocess_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>image_path,<br>segmentation_path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L134'>CocoBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoImagesAndCaptionsTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L153'>CocoImagesAndCaptionsTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>onehot_segmentation = False,<br>use_stuffthing = False,<br>crop_size = None,<br>force_no_crop = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoImagesAndCaptionsTrain:get_split' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L160'>CocoImagesAndCaptionsTrain:get_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoImagesAndCaptionsValidation:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L166'>CocoImagesAndCaptionsValidation:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>onehot_segmentation = False,<br>use_stuffthing = False,<br>crop_size = None,<br>force_no_crop = False,<br>given_files = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/coco.py:CocoImagesAndCaptionsValidation:get_split' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/coco.py#L160'>CocoImagesAndCaptionsValidation:get_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>    """returns a pair of (image, caption)"""
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='main.py' href='https://github.com/CompVis/taming-transformers/tree/master/main.py'>main.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='main.py:get_obj_from_str' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L14'>get_obj_from_str</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>string,<br>reload = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='main.py:get_parser' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L22'>get_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>**parser_kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='main.py:nondefault_trainer_args' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L106'>nondefault_trainer_args</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>opt,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='main.py:instantiate_from_config' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L113'>instantiate_from_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='main.py:WrappedDataset' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L119'>WrappedDataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='main.py:DataModuleFromConfig' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L131'>DataModuleFromConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='main.py:SetupCallback' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L175'>SetupCallback</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='main.py:ImageLogger' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L215'>ImageLogger</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:WrappedDataset:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L121'>WrappedDataset:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dataset,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:WrappedDataset:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L124'>WrappedDataset:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:WrappedDataset:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L127'>WrappedDataset:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L132'>DataModuleFromConfig:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch_size,<br>train = None,<br>validation = None,<br>test = None,<br>wrap = False,<br>num_workers = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:prepare_data' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L149'>DataModuleFromConfig:prepare_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:setup' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L153'>DataModuleFromConfig:setup</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>stage = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:_train_dataloader' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L161'>DataModuleFromConfig:_train_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:_val_dataloader' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L165'>DataModuleFromConfig:_val_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:DataModuleFromConfig:_test_dataloader' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L170'>DataModuleFromConfig:_test_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:SetupCallback:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L176'>SetupCallback:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>resume,<br>now,<br>logdir,<br>ckptdir,<br>cfgdir,<br>config,<br>lightning_config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:SetupCallback:on_pretrain_routine_start' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L186'>SetupCallback:on_pretrain_routine_start</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>trainer,<br>pl_module,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L216'>ImageLogger:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch_frequency,<br>max_images,<br>clamp = True,<br>increase_log_steps = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:_wandb' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L230'>ImageLogger:_wandb</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pl_module,<br>images,<br>batch_idx,<br>split,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:_testtube' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L239'>ImageLogger:_testtube</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pl_module,<br>images,<br>batch_idx,<br>split,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:log_local' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L250'>ImageLogger:log_local</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>save_dir,<br>split,<br>images,<br>global_step,<br>current_epoch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:log_img' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L269'>ImageLogger:log_img</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pl_module,<br>batch,<br>batch_idx,<br>split = "train",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:check_frequency' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L300'>ImageLogger:check_frequency</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:on_train_batch_end' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L309'>ImageLogger:on_train_batch_end</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>trainer,<br>pl_module,<br>outputs,<br>batch,<br>batch_idx,<br>dataloader_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='main.py:ImageLogger:on_validation_batch_end' href='https://github.com/CompVis/taming-transformers/tree/master/main.py#L312'>ImageLogger:on_validation_batch_end</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>trainer,<br>pl_module,<br>outputs,<br>batch,<br>batch_idx,<br>dataloader_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/ade20k.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py'>taming/data/ade20k.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/ade20k.py:Examples' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L11'>Examples</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/ade20k.py:ADE20kBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L22'>ADE20kBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/ade20k.py:ADE20kTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L101'>ADE20kTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/ade20k.py:ADE20kValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L111'>ADE20kValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:Examples:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L12'>Examples:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size = 256,<br>random_crop = False,<br>interpolation = "bicubic",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L23'>ADE20kBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br>size = None,<br>random_crop = False,<br>interpolation = "bicubic",<br>crop_size = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L75'>ADE20kBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L78'>ADE20kBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L103'>ADE20kTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br>size = None,<br>random_crop = True,<br>interpolation = "bicubic",<br>crop_size = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kTrain:get_split' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L107'>ADE20kTrain:get_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/ade20k.py:ADE20kValidation:get_split' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/ade20k.py#L107'>ADE20kValidation:get_split</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/vqvae/quantize.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py'>taming/modules/vqvae/quantize.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L9'>VectorQuantizer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L110'>GumbelQuantize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L213'>VectorQuantizer2</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L25'>VectorQuantizer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n_e,<br>e_dim,<br>beta,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L34'>VectorQuantizer:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>z,<br></ul>
        <li>Docs:<br>        """
<br>
        Inputs the output of the encoder network z and maps it to a discrete
<br>
        one-hot vector that is the index of the closest embedding vector e_j
<br>
        z (continuous) -> z_q (discrete)
<br>
        z.shape = (batch, channel, height, width)
<br>
        quantization pipeline:
<br>
            1. get encoder input (B,C,H,W)
<br>
            2. flatten input to (B*H*W,C)
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer:get_codebook_entry' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L92'>VectorQuantizer:get_codebook_entry</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>indices,<br>shape,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L117'>GumbelQuantize:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>num_hiddens,<br>embedding_dim,<br>n_embed,<br>straight_through = True,<br>kl_weight = 5e-4,<br>temp_init = 1.0,<br>use_vqinterface = True,<br>remap = None,<br>unknown_index = "random",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize:remap_to_used' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L147'>GumbelQuantize:remap_to_used</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>inds,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize:unmap_to_all' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L161'>GumbelQuantize:unmap_to_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>inds,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L171'>GumbelQuantize:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>z,<br>temp = None,<br>return_logits = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:GumbelQuantize:get_codebook_entry' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L92'>GumbelQuantize:get_codebook_entry</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>indices,<br>shape,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L221'>VectorQuantizer2:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n_e,<br>e_dim,<br>beta,<br>remap = None,<br>unknown_index = "random",<br>sane_index_shape = False,<br>legacy = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2:remap_to_used' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L147'>VectorQuantizer2:remap_to_used</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>inds,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2:unmap_to_all' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L161'>VectorQuantizer2:unmap_to_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>inds,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L271'>VectorQuantizer2:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>z,<br>temp = None,<br>rescale_logits = False,<br>return_logits = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/vqvae/quantize.py:VectorQuantizer2:get_codebook_entry' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/vqvae/quantize.py#L92'>VectorQuantizer2:get_codebook_entry</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>indices,<br>shape,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/losses/vqperceptual.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py'>taming/modules/losses/vqperceptual.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/losses/vqperceptual.py:adopt_weight' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L14'>adopt_weight</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>weight,<br>global_step,<br>threshold = 0,<br>value = 0.,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/losses/vqperceptual.py:hinge_d_loss' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L20'>hinge_d_loss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logits_real,<br>logits_fake,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/losses/vqperceptual.py:vanilla_d_loss' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L27'>vanilla_d_loss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logits_real,<br>logits_fake,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/vqperceptual.py:DummyLoss' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L9'>DummyLoss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/vqperceptual.py:VQLPIPSWithDiscriminator' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L34'>VQLPIPSWithDiscriminator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/vqperceptual.py:DummyLoss:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L10'>DummyLoss:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/vqperceptual.py:VQLPIPSWithDiscriminator:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L35'>VQLPIPSWithDiscriminator:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>disc_start,<br>codebook_weight = 1.0,<br>pixelloss_weight = 1.0,<br>disc_num_layers = 3,<br>disc_in_channels = 3,<br>disc_factor = 1.0,<br>disc_weight = 1.0,<br>perceptual_weight = 1.0,<br>use_actnorm = False,<br>disc_conditional = False,<br>disc_ndf = 64,<br>disc_loss = "hinge",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/vqperceptual.py:VQLPIPSWithDiscriminator:calculate_adaptive_weight' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L63'>VQLPIPSWithDiscriminator:calculate_adaptive_weight</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>nll_loss,<br>g_loss,<br>last_layer = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/vqperceptual.py:VQLPIPSWithDiscriminator:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/vqperceptual.py#L76'>VQLPIPSWithDiscriminator:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>codebook_loss,<br>inputs,<br>reconstructions,<br>optimizer_idx,<br>global_step,<br>last_layer = None,<br>cond = None,<br>split = "train",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='scripts/make_samples.py' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py'>scripts/make_samples.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:save_image' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L12'>save_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:run_conditional' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L20'>run_conditional</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>dsets,<br>outdir,<br>top_k,<br>temperature,<br>batch_size = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:get_parser' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L124'>get_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:load_model_from_config' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L179'>load_model_from_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>sd,<br>gpu = True,<br>eval_mode = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:get_data' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L209'>get_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/make_samples.py:load_model_and_dset' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/make_samples.py#L217'>load_model_and_dset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>ckpt,<br>gpu,<br>eval_mode,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/models/cond_transformer.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py'>taming/models/cond_transformer.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/models/cond_transformer.py:disabled_train' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L10'>disabled_train</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>mode = True,<br></ul>
        <li>Docs:<br>    """Overwrite model.train with this function to make sure train/eval mode
<br>
    does not change anymore."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/models/cond_transformer.py:Net2NetTransformer' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L16'>Net2NetTransformer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L17'>Net2NetTransformer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>transformer_config,<br>first_stage_config,<br>cond_stage_config,<br>permuter_config = None,<br>ckpt_path = None,<br>ignore_keys = [],<br>first_stage_key = "image",<br>cond_stage_key = "depth",<br>downsample_cond_size = -1,<br>pkeep = 1.0,<br>sos_token = 0,<br>unconditional = False,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:init_from_ckpt' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L48'>Net2NetTransformer:init_from_ckpt</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br>ignore_keys = list(,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:init_first_stage_from_ckpt' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L58'>Net2NetTransformer:init_first_stage_from_ckpt</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:init_cond_stage_from_ckpt' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L64'>Net2NetTransformer:init_cond_stage_from_ckpt</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L80'>Net2NetTransformer:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>c,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:top_k_logits' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L106'>Net2NetTransformer:top_k_logits</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>logits,<br>k,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:sample' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L113'>Net2NetTransformer:sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>c,<br>steps,<br>temperature = 1.0,<br>sample = False,<br>top_k = None,<br>callback=lambda k:  = lambda k: None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:encode_to_z' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L169'>Net2NetTransformer:encode_to_z</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:encode_to_c' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L176'>Net2NetTransformer:encode_to_c</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>c,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:decode_to_img' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L185'>Net2NetTransformer:decode_to_img</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>index,<br>zshape,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:log_images' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L194'>Net2NetTransformer:log_images</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>temperature = None,<br>top_k = None,<br>callback = None,<br>lr_interface = False,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:get_input' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L265'>Net2NetTransformer:get_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>key,<br>batch,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:get_xc' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L275'>Net2NetTransformer:get_xc</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>N = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:shared_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L283'>Net2NetTransformer:shared_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:training_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L289'>Net2NetTransformer:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:validation_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L294'>Net2NetTransformer:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/cond_transformer.py:Net2NetTransformer:configure_optimizers' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/cond_transformer.py#L299'>Net2NetTransformer:configure_optimizers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """
<br>
        Following minGPT:
<br>
        This long function is unfortunately doing something very simple and is being very defensive:
<br>
        We are separating out all parameters of the model into two buckets: those that will experience
<br>
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
<br>
        We are then returning the PyTorch optimizer object.
<br>
        """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='scripts/sample_fast.py' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py'>scripts/sample_fast.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:chw_to_pillow' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L17'>chw_to_pillow</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:sample_classconditional' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L22'>sample_classconditional</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>batch_size,<br>class_label,<br>steps = 256,<br>temperature = None,<br>top_k = None,<br>callback = None,<br>dim_z = 256,<br>h = 16,<br>w = 16,<br>verbose_time = False,<br>top_p = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:sample_unconditional' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L43'>sample_unconditional</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>batch_size,<br>steps = 256,<br>temperature = None,<br>top_k = None,<br>top_p = None,<br>callback = None,<br>dim_z = 256,<br>h = 16,<br>w = 16,<br>verbose_time = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:run' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L62'>run</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logdir,<br>model,<br>batch_size,<br>temperature,<br>top_k,<br>unconditional = True,<br>num_samples = 50000,<br>given_classes = None,<br>top_p = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:save_from_logs' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L84'>save_from_logs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logs,<br>logdir,<br>base_count,<br>key = "samples",<br>cond_key = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:get_parser' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L98'>get_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:load_model_from_config' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L183'>load_model_from_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>sd,<br>gpu = True,<br>eval_mode = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_fast.py:load_model' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_fast.py#L194'>load_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>sd,<br>gpu = True,<br>eval_mode = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/transformer/mingpt.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py'>taming/modules/transformer/mingpt.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/transformer/mingpt.py:top_k_logits' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L286'>top_k_logits</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logits,<br>k,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/transformer/mingpt.py:sample' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L293'>sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>x,<br>steps,<br>temperature = 1.0,<br>sample = False,<br>top_k = None,<br></ul>
        <li>Docs:<br>    """
<br>
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
<br>
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
<br>
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
<br>
    of block_size, unlike an RNN that has an infinite context window.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/transformer/mingpt.py:sample_with_past' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L324'>sample_with_past</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>model,<br>steps,<br>temperature = 1.,<br>sample_logits = True,<br>top_k = None,<br>top_p = None,<br>callback = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:GPTConfig' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L22'>GPTConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:GPT1Config' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L35'>GPT1Config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:CausalSelfAttention' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L42'>CausalSelfAttention</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:Block' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L98'>Block</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:GPT' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L125'>GPT</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:DummyGPT' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L215'>DummyGPT</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:CodeGPT' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L225'>CodeGPT</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/mingpt.py:KMeans' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L356'>KMeans</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPTConfig:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L28'>GPTConfig:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>vocab_size,<br>block_size,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CausalSelfAttention:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L49'>CausalSelfAttention:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CausalSelfAttention:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L69'>CausalSelfAttention:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>layer_past = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:Block:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L49'>Block:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:Block:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L112'>Block:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>layer_past = None,<br>return_present = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPT:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L127'>GPT:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>vocab_size,<br>block_size,<br>n_layer = 12,<br>n_head = 8,<br>n_embd = 256,<br>embd_pdrop = 0.,<br>resid_pdrop = 0.,<br>attn_pdrop = 0.,<br>n_unmasked = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPT:get_block_size' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L148'>GPT:get_block_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPT:_init_weights' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L151'>GPT:_init_weights</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>module,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPT:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L160'>GPT:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br>embeddings = None,<br>targets = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:GPT:forward_with_past' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L182'>GPT:forward_with_past</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br>embeddings = None,<br>targets = None,<br>past = None,<br>past_length = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:DummyGPT:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L217'>DummyGPT:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>add_value = 1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:DummyGPT:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L221'>DummyGPT:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CodeGPT:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L227'>CodeGPT:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>vocab_size,<br>block_size,<br>in_channels,<br>n_layer = 12,<br>n_head = 8,<br>n_embd = 256,<br>embd_pdrop = 0.,<br>resid_pdrop = 0.,<br>attn_pdrop = 0.,<br>n_unmasked = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CodeGPT:get_block_size' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L148'>CodeGPT:get_block_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CodeGPT:_init_weights' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L151'>CodeGPT:_init_weights</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>module,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:CodeGPT:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L160'>CodeGPT:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br>embeddings = None,<br>targets = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:KMeans:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L357'>KMeans:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ncluster = 512,<br>nc = 3,<br>niter = 10,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:KMeans:is_initialized' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L366'>KMeans:is_initialized</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:KMeans:initialize' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L370'>KMeans:initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/mingpt.py:KMeans:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/mingpt.py#L389'>KMeans:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br>shape = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/imagenet.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py'>taming/data/imagenet.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/data/imagenet.py:give_synsets_from_indices' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L15'>give_synsets_from_indices</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>indices,<br>path_to_yaml = "data/imagenet_idx_to_synset.yaml",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/imagenet.py:str_to_indices' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L25'>str_to_indices</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>string,<br></ul>
        <li>Docs:<br>    """Expects a string in the format '32-123, 256, 280-321'"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/imagenet.py:get_preprocessor' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L244'>get_preprocessor</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>size = None,<br>random_crop = False,<br>additional_targets = None,<br>crop_size = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/imagenet.py:rgba_to_depth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L273'>rgba_to_depth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/imagenet.py:imscale' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L416'>imscale</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>factor,<br>keepshapes = False,<br>keepmode = "bicubic",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L41'>ImageNetBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L123'>ImageNetTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L178'>ImageNetValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:BaseWithDepth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L282'>BaseWithDepth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetTrainWithDepth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L328'>ImageNetTrainWithDepth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetValidationWithDepth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L346'>ImageNetValidationWithDepth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:RINTrainWithDepth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L363'>RINTrainWithDepth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:RINValidationWithDepth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L370'>RINValidationWithDepth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:DRINExamples' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L377'>DRINExamples</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetScale' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L446'>ImageNetScale</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetScaleTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L510'>ImageNetScaleTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetScaleValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L517'>ImageNetScaleValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetEdges' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L526'>ImageNetEdges</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetEdgesTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L549'>ImageNetEdgesTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/imagenet.py:ImageNetEdgesValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L556'>ImageNetEdgesValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L42'>ImageNetBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L51'>ImageNetBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L54'>ImageNetBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:_prepare' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L57'>ImageNetBase:_prepare</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:_filter_relpaths' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L60'>ImageNetBase:_filter_relpaths</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>relpaths,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:_prepare_synset_to_human' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L77'>ImageNetBase:_prepare_synset_to_human</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:_prepare_idx_to_synset' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L85'>ImageNetBase:_prepare_idx_to_synset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetBase:_load' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L91'>ImageNetBase:_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetTrain:_prepare' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L57'>ImageNetTrain:_prepare</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetValidation:_prepare' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L57'>ImageNetValidation:_prepare</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:BaseWithDepth:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L285'>BaseWithDepth:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br>size = None,<br>random_crop = False,<br>crop_size = None,<br>root = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:BaseWithDepth:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L51'>BaseWithDepth:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:BaseWithDepth:preprocess_depth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L305'>BaseWithDepth:preprocess_depth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:BaseWithDepth:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L54'>BaseWithDepth:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetTrainWithDepth:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L330'>ImageNetTrainWithDepth:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>random_crop = True,<br>sub_indices = None,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetTrainWithDepth:get_base_dset' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L334'>ImageNetTrainWithDepth:get_base_dset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetTrainWithDepth:get_depth_path' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L340'>ImageNetTrainWithDepth:get_depth_path</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>e,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetValidationWithDepth:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L347'>ImageNetValidationWithDepth:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>sub_indices = None,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetValidationWithDepth:get_base_dset' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L334'>ImageNetValidationWithDepth:get_base_dset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetValidationWithDepth:get_depth_path' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L340'>ImageNetValidationWithDepth:get_depth_path</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>e,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:RINTrainWithDepth:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L364'>RINTrainWithDepth:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br>size = None,<br>random_crop = True,<br>crop_size = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:RINValidationWithDepth:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L371'>RINValidationWithDepth:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config = None,<br>size = None,<br>random_crop = False,<br>crop_size = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:DRINExamples:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L378'>DRINExamples:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:DRINExamples:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L51'>DRINExamples:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:DRINExamples:preprocess_image' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L390'>DRINExamples:preprocess_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>image_path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:DRINExamples:preprocess_depth' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L305'>DRINExamples:preprocess_depth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:DRINExamples:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L54'>DRINExamples:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScale:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L447'>ImageNetScale:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size = None,<br>crop_size = None,<br>random_crop = False,<br>up_factor = None,<br>hr_factor = None,<br>keep_mode = "bicubic",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScale:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L51'>ImageNetScale:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScale:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L54'>ImageNetScale:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScaleTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L511'>ImageNetScaleTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>random_crop = True,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScaleTrain:get_base' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L514'>ImageNetScaleTrain:get_base</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetScaleValidation:get_base' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L514'>ImageNetScaleValidation:get_base</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetEdges:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L527'>ImageNetEdges:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>up_factor = 1,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetEdges:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L54'>ImageNetEdges:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetEdgesTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L511'>ImageNetEdgesTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>random_crop = True,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetEdgesTrain:get_base' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L514'>ImageNetEdgesTrain:get_base</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/imagenet.py:ImageNetEdgesValidation:get_base' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/imagenet.py#L514'>ImageNetEdgesValidation:get_base</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/transformer/permuter.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py'>taming/modules/transformer/permuter.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/transformer/permuter.py:mortonify' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L47'>mortonify</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>i,<br>j,<br></ul>
        <li>Docs:<br>    """(i,j) index to linear morton code"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:AbstractPermuter' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L6'>AbstractPermuter</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:Identity' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L13'>Identity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:Subsample' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L21'>Subsample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:ZCurve' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L62'>ZCurve</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:SpiralOut' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L81'>SpiralOut</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:SpiralIn' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L141'>SpiralIn</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:Random' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L202'>Random</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/transformer/permuter.py:AlternateParsing' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L217'>AlternateParsing</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:AbstractPermuter:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L7'>AbstractPermuter:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:AbstractPermuter:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>AbstractPermuter:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Identity:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L14'>Identity:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Identity:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>Identity:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Subsample:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>Subsample:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Subsample:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>Subsample:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:ZCurve:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>ZCurve:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:ZCurve:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>ZCurve:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:SpiralOut:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>SpiralOut:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:SpiralOut:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>SpiralOut:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:SpiralIn:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>SpiralIn:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:SpiralIn:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>SpiralIn:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Random:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>Random:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:Random:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>Random:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:AlternateParsing:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L22'>AlternateParsing:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>H,<br>W,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/transformer/permuter.py:AlternateParsing:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/transformer/permuter.py#L9'>AlternateParsing:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/util.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py'>taming/util.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/util.py:download' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L18'>download</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>url,<br>local_path,<br>chunk_size = 1024,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/util.py:md5_hash' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L30'>md5_hash</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/util.py:get_ckpt_path' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L36'>get_ckpt_path</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name,<br>root,<br>check = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/util.py:retrieve' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L62'>retrieve</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>list_or_dict,<br>key,<br>splitval = "/",<br>default = None,<br>expand = True,<br>pass_success = False,<br></ul>
        <li>Docs:<br>    """Given a nested list or dict return the desired value at key expanding
<br>
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
<br>
    is done in-place.
<br>

<br>
    Parameters
<br>
    ----------
<br>
        list_or_dict : list or dict
<br>
            Possibly nested list or dictionary.
<br>
        key : str
<br>
            key/to/value, path like string describing all keys necessary to
<br>
            consider to get to the desired value. List indices can also be
<br>
            passed here.
<br>
        splitval : str
<br>
            String that defines the delimiter between keys of the
<br>
            different depth levels in `key`.
<br>
        default : obj
<br>
            Value returned if :attr:`key` is not found.
<br>
        expand : bool
<br>
            Whether to expand callable nodes on the path or not.
<br>

<br>
    Returns
<br>
    -------
<br>
        The desired value or if :attr:`default` is not ``None`` and the
<br>
        :attr:`key` is not found returns ``default``.
<br>

<br>
    Raises
<br>
    ------
<br>
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
<br>
        ``None``.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/util.py:KeyNotFoundError' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L47'>KeyNotFoundError</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/util.py:KeyNotFoundError:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/util.py#L48'>KeyNotFoundError:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>cause,<br>keys = None,<br>visited = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/faceshq.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py'>taming/data/faceshq.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:FacesBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L9'>FacesBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:CelebAHQTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L29'>CelebAHQTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:CelebAHQValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L40'>CelebAHQValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:FFHQTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L51'>FFHQTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:FFHQValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L62'>FFHQValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:FacesHQTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L73'>FacesHQTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/faceshq.py:FacesHQValidation' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L105'>FacesHQValidation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L10'>FacesBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L15'>FacesBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L18'>FacesBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:CelebAHQTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L30'>CelebAHQTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:CelebAHQValidation:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L30'>CelebAHQValidation:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FFHQTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L30'>FFHQTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FFHQValidation:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L30'>FFHQValidation:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L75'>FacesHQTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br>crop_size = None,<br>coord = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQTrain:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L15'>FacesHQTrain:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQTrain:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L18'>FacesHQTrain:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQValidation:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L75'>FacesHQValidation:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>keys = None,<br>crop_size = None,<br>coord = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQValidation:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L15'>FacesHQValidation:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/faceshq.py:FacesHQValidation:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/faceshq.py#L18'>FacesHQValidation:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/losses/segmentation.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py'>taming/modules/losses/segmentation.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/modules/losses/segmentation.py:BCELoss' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py#L5'>BCELoss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/losses/segmentation.py:BCELossWithQuant' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py#L11'>BCELossWithQuant</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/segmentation.py:BCELoss:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py#L6'>BCELoss:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>prediction,<br>target,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/segmentation.py:BCELossWithQuant:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py#L12'>BCELossWithQuant:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>codebook_weight = 1.,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/losses/segmentation.py:BCELossWithQuant:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/losses/segmentation.py#L16'>BCELossWithQuant:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>qloss,<br>target,<br>prediction,<br>split,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/custom.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py'>taming/data/custom.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/custom.py:CustomBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L9'>CustomBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/custom.py:CustomTrain' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L23'>CustomTrain</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/custom.py:CustomTest' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L31'>CustomTest</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/custom.py:CustomBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L10'>CustomBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/custom.py:CustomBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L14'>CustomBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/custom.py:CustomBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L17'>CustomBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/custom.py:CustomTrain:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L24'>CustomTrain:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>training_images_list_file,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/custom.py:CustomTest:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/custom.py#L32'>CustomTest:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size,<br>test_images_list_file,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='scripts/sample_conditional.py' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py'>scripts/sample_conditional.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:bchw_to_st' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L16'>bchw_to_st</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:save_img' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L19'>save_img</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>xstart,<br>fname,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:get_interactive_image' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L25'>get_interactive_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>resize = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:single_image_to_torch' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L40'>single_image_to_torch</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>permute = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:pad_to_M' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L49'>pad_to_M</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>M,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:run_conditional' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L56'>run_conditional</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>dsets,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:get_parser' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L205'>get_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:load_model_from_config' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L242'>load_model_from_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>sd,<br>gpu = True,<br>eval_mode = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:get_data' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L272'>get_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/sample_conditional.py:load_model_and_dset' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/sample_conditional.py#L281'>load_model_and_dset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br>ckpt,<br>gpu,<br>eval_mode,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/base.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py'>taming/data/base.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/base.py:ConcatDatasetWithIndex' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L8'>ConcatDatasetWithIndex</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/base.py:ImagePaths' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L23'>ImagePaths</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/base.py:NumpyPaths' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L62'>NumpyPaths</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:ConcatDatasetWithIndex:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L10'>ConcatDatasetWithIndex:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:ImagePaths:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L24'>ImagePaths:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>paths,<br>size = None,<br>random_crop = False,<br>labels = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:ImagePaths:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L42'>ImagePaths:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:ImagePaths:preprocess_image' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L45'>ImagePaths:preprocess_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>image_path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:ImagePaths:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L54'>ImagePaths:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/base.py:NumpyPaths:preprocess_image' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/base.py#L45'>NumpyPaths:preprocess_image</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>image_path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/util.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py'>taming/modules/util.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/util.py:count_params' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L5'>count_params</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/util.py:ActNorm' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L10'>ActNorm</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/util.py:AbstractEncoder' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L95'>AbstractEncoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/util.py:Labelator' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L103'>Labelator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/util.py:SOSProvider' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L117'>SOSProvider</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:ActNorm:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L11'>ActNorm:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>num_features,<br>logdet = False,<br>affine = True,<br>allow_reverse_init = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:ActNorm:initialize' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L22'>ActNorm:initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:ActNorm:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L43'>ActNorm:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br>reverse = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:ActNorm:reverse' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L71'>ActNorm:reverse</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>output,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:AbstractEncoder:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L96'>AbstractEncoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:AbstractEncoder:encode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L99'>AbstractEncoder:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:Labelator:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L105'>Labelator:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n_classes,<br>quantize_interface = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:Labelator:encode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L110'>Labelator:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>c,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:SOSProvider:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L119'>SOSProvider:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>sos_token,<br>quantize_interface = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/util.py:SOSProvider:encode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/util.py#L124'>SOSProvider:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='scripts/extract_depth.py' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py'>scripts/extract_depth.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:get_state' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L8'>get_state</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>gpu,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:depth_to_rgba' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L23'>depth_to_rgba</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:rgba_to_depth' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L32'>rgba_to_depth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:run' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L41'>run</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>state,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:get_filename' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L57'>get_filename</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>relpath,<br>level = -2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='scripts/extract_depth.py:save_depth' href='https://github.com/CompVis/taming-transformers/tree/master/scripts/extract_depth.py#L65'>save_depth</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>dataset,<br>path,<br>debug = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/utils.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py'>taming/data/utils.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:unpack' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L9'>unpack</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:reporthook' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L25'>reporthook</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>bar,<br></ul>
        <li>Docs:<br>    """tqdm progress bar for downloads."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:get_root' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L36'>get_root</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>name,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:is_prepared' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L43'>is_prepared</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>root,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:mark_prepared' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L47'>mark_prepared</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>root,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:prompt_download' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L51'>prompt_download</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_,<br>source,<br>target_dir,<br>content_dir = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:download_url' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L71'>download_url</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>file_,<br>url,<br>target_dir,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:download_urls' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L81'>download_urls</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>urls,<br>target_dir,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/data/utils.py:quadratic_crop' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/utils.py#L89'>quadratic_crop</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>bbox,<br>alpha = 1.0,<br></ul>
        <li>Docs:<br>    """bbox is xmin, ymin, xmax, ymax"""
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/lr_scheduler.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/lr_scheduler.py'>taming/lr_scheduler.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/lr_scheduler.py:LambdaWarmUpCosineScheduler' href='https://github.com/CompVis/taming-transformers/tree/master/taming/lr_scheduler.py#L4'>LambdaWarmUpCosineScheduler</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/lr_scheduler.py:LambdaWarmUpCosineScheduler:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/lr_scheduler.py#L8'>LambdaWarmUpCosineScheduler:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>warm_up_steps,<br>lr_min,<br>lr_max,<br>lr_start,<br>max_decay_steps,<br>verbosity_interval = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/lr_scheduler.py:LambdaWarmUpCosineScheduler:schedule' href='https://github.com/CompVis/taming-transformers/tree/master/taming/lr_scheduler.py#L17'>LambdaWarmUpCosineScheduler:schedule</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/lr_scheduler.py:LambdaWarmUpCosineScheduler:__call__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/lr_scheduler.py#L32'>LambdaWarmUpCosineScheduler:__call__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/models/vqgan.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py'>taming/models/vqgan.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/models/vqgan.py:VQModel' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L12'>VQModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/models/vqgan.py:VQSegmentationModel' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L159'>VQSegmentationModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/models/vqgan.py:VQNoDiscModel' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L211'>VQNoDiscModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/models/vqgan.py:GumbelVQ' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L261'>GumbelVQ</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L13'>VQModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ddconfig,<br>lossconfig,<br>n_embed,<br>embed_dim,<br>ckpt_path = None,<br>ignore_keys = [],<br>image_key = "image",<br>colorize_nlabels = None,<br>monitor = None,<br>remap = None,<br>sane_index_shape = False,<br># tell vector quantizer to return indices as bhw,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:init_from_ckpt' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L44'>VQModel:init_from_ckpt</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br>ignore_keys = list(,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:encode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L55'>VQModel:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:decode' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L61'>VQModel:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>quant,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:decode_code' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L66'>VQModel:decode_code</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>code_b,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L71'>VQModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:get_input' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L76'>VQModel:get_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>k,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:training_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L83'>VQModel:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br>optimizer_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:validation_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L104'>VQModel:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:configure_optimizers' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L121'>VQModel:configure_optimizers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:get_last_layer' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L133'>VQModel:get_last_layer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:log_images' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L136'>VQModel:log_images</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQModel:to_rgb' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L150'>VQModel:to_rgb</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQSegmentationModel:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L160'>VQSegmentationModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>n_labels,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQSegmentationModel:configure_optimizers' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L121'>VQSegmentationModel:configure_optimizers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQSegmentationModel:training_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L174'>VQSegmentationModel:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQSegmentationModel:validation_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L104'>VQSegmentationModel:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQSegmentationModel:log_images' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L136'>VQSegmentationModel:log_images</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQNoDiscModel:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L13'>VQNoDiscModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ddconfig,<br>lossconfig,<br>n_embed,<br>embed_dim,<br>ckpt_path = None,<br>ignore_keys = [],<br>image_key = "image",<br>colorize_nlabels = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQNoDiscModel:training_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L174'>VQNoDiscModel:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQNoDiscModel:validation_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L104'>VQNoDiscModel:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:VQNoDiscModel:configure_optimizers' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L121'>VQNoDiscModel:configure_optimizers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L13'>GumbelVQ:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ddconfig,<br>lossconfig,<br>n_embed,<br>embed_dim,<br>temperature_scheduler_config,<br>ckpt_path = None,<br>ignore_keys = [],<br>image_key = "image",<br>colorize_nlabels = None,<br>monitor = None,<br>kl_weight = 1e-8,<br>remap = None,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:temperature_scheduling' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L302'>GumbelVQ:temperature_scheduling</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:encode_to_prequant' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L305'>GumbelVQ:encode_to_prequant</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:decode_code' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L66'>GumbelVQ:decode_code</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>code_b,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:training_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L83'>GumbelVQ:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br>optimizer_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:validation_step' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L104'>GumbelVQ:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/models/vqgan.py:GumbelVQ:log_images' href='https://github.com/CompVis/taming-transformers/tree/master/taming/models/vqgan.py#L136'>GumbelVQ:log_images</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/discriminator/model.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/discriminator/model.py'>taming/modules/discriminator/model.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/discriminator/model.py:weights_init' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/discriminator/model.py#L8'>weights_init</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>m,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/discriminator/model.py:NLayerDiscriminator' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/discriminator/model.py#L17'>NLayerDiscriminator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/discriminator/model.py:NLayerDiscriminator:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/discriminator/model.py#L21'>NLayerDiscriminator:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input_nc = 3,<br>ndf = 64,<br>n_layers = 3,<br>use_actnorm = False,<br></ul>
        <li>Docs:<br>        """Construct a PatchGAN discriminator
<br>
        Parameters:
<br>
            input_nc (int)  -- the number of channels in input images
<br>
            ndf (int)       -- the number of filters in the last conv layer
<br>
            n_layers (int)  -- the number of conv layers in the discriminator
<br>
            norm_layer      -- normalization layer
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/discriminator/model.py:NLayerDiscriminator:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/discriminator/model.py#L65'>NLayerDiscriminator:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br></ul>
        <li>Docs:<br>        """Standard forward."""
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/data/sflckr.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py'>taming/data/sflckr.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='taming/data/sflckr.py:SegmentationBase' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L9'>SegmentationBase</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/data/sflckr.py:Examples' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L86'>Examples</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/sflckr.py:SegmentationBase:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L10'>SegmentationBase:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>data_csv,<br>data_root,<br>segmentation_root,<br>size = None,<br>random_crop = False,<br>interpolation = "bicubic",<br>n_labels = 182,<br>shift_segmentation = False,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/sflckr.py:SegmentationBase:__len__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L52'>SegmentationBase:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/sflckr.py:SegmentationBase:__getitem__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L55'>SegmentationBase:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>i,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/data/sflckr.py:Examples:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/data/sflckr.py#L87'>Examples:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>size = None,<br>random_crop = False,<br>interpolation = "bicubic",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='taming/modules/diffusionmodules/model.py' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py'>taming/modules/diffusionmodules/model.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='taming/modules/diffusionmodules/model.py:get_timestep_embedding' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L8'>get_timestep_embedding</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>timesteps,<br>embedding_dim,<br></ul>
        <li>Docs:<br>    """
<br>
    This matches the implementation in Denoising Diffusion Probabilistic Models:
<br>
    From Fairseq.
<br>
    Build sinusoidal embeddings.
<br>
    This matches the implementation in tensor2tensor, but differs slightly
<br>
    from the description in Section 3.5 of "Attention Is All You Need".
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/diffusionmodules/model.py:nonlinearity' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L29'>nonlinearity</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='taming/modules/diffusionmodules/model.py:Normalize' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L34'>Normalize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>in_channels,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:Upsample' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L38'>Upsample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:Downsample' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L56'>Downsample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:ResnetBlock' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L78'>ResnetBlock</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:AttnBlock' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L140'>AttnBlock</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:Model' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L195'>Model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:Encoder' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L342'>Encoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:Decoder' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L436'>Decoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:VUNet' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L540'>VUNet</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:SimpleDecoder' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L694'>SimpleDecoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='taming/modules/diffusionmodules/model.py:UpsampleDecoder' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L730'>UpsampleDecoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Upsample:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L39'>Upsample:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_channels,<br>with_conv,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Upsample:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>Upsample:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Downsample:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L39'>Downsample:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_channels,<br>with_conv,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Downsample:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>Downsample:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:ResnetBlock:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L79'>ResnetBlock:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>in_channels,<br>out_channels = None,<br>conv_shortcut = False,<br>dropout,<br>temb_channels = 512,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:ResnetBlock:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L117'>ResnetBlock:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>temb,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:AttnBlock:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L141'>AttnBlock:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_channels,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:AttnBlock:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>AttnBlock:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Model:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L196'>Model:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>ch,<br>out_ch,<br>ch_mult = (1,<br>2,<br>4,<br>8),<br>num_res_blocks,<br>attn_resolutions,<br>dropout = 0.0,<br>resamp_with_conv = True,<br>in_channels,<br>resolution,<br>use_timestep = True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Model:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L295'>Model:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>t = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Encoder:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L196'>Encoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>ch,<br>out_ch,<br>ch_mult = (1,<br>2,<br>4,<br>8),<br>num_res_blocks,<br>attn_resolutions,<br>dropout = 0.0,<br>resamp_with_conv = True,<br>in_channels,<br>resolution,<br>z_channels,<br>double_z = True,<br>**ignore_kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Encoder:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>Encoder:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Decoder:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L196'>Decoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>ch,<br>out_ch,<br>ch_mult = (1,<br>2,<br>4,<br>8),<br>num_res_blocks,<br>attn_resolutions,<br>dropout = 0.0,<br>resamp_with_conv = True,<br>in_channels,<br>resolution,<br>z_channels,<br>give_pre_end = False,<br>**ignorekwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:Decoder:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L506'>Decoder:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>z,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:VUNet:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L196'>VUNet:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>ch,<br>out_ch,<br>ch_mult = (1,<br>2,<br>4,<br>8),<br>num_res_blocks,<br>attn_resolutions,<br>dropout = 0.0,<br>resamp_with_conv = True,<br>in_channels,<br>c_channels,<br>resolution,<br>z_channels,<br>use_timestep = False,<br>**ignore_kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:VUNet:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L645'>VUNet:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>z,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:SimpleDecoder:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L695'>SimpleDecoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_channels,<br>out_channels,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:SimpleDecoder:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>SimpleDecoder:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:UpsampleDecoder:__init__' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L731'>UpsampleDecoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_channels,<br>out_channels,<br>ch,<br>num_res_blocks,<br>resolution,<br>2,<br>2),<br>dropout=0.0):  = 0.0):,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='taming/modules/diffusionmodules/model.py:UpsampleDecoder:forward' href='https://github.com/CompVis/taming-transformers/tree/master/taming/modules/diffusionmodules/model.py#L49'>UpsampleDecoder:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>
